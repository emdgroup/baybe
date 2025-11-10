"""Transformation utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np
from attrs import evolve

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.transformations.base import Transformation

    TensorCallable = Callable[[Tensor], Tensor]
    """Type alias for a torch-based function mapping from reals to reals."""


def convert_transformation(x: Transformation | TensorCallable, /) -> Transformation:
    """Autowrap a torch callable as transformation (with transformation passthrough)."""
    from baybe.transformations.base import Transformation
    from baybe.transformations.basic import CustomTransformation

    return x if isinstance(x, Transformation) else CustomTransformation(x)


def combine_affine_transformations(t1, t2, /):
    """Combine two affine transformations into one."""
    from baybe.transformations.basic import AffineTransformation

    factor = t2.factor * t1.factor
    shift = t2.factor * t1.shift + t2.shift

    if not np.all(np.isfinite([factor, shift])):
        raise OverflowError(
            "The combined affine transformation produces infinite values."
        )

    return AffineTransformation(factor=factor, shift=shift)


def _flatten_transformations(
    transformations: Iterable[Transformation], /
) -> Iterable[Transformation]:
    """Recursively flatten nested chained transformations."""
    from baybe.transformations.composite import ChainedTransformation

    for t in transformations:
        if isinstance(t, ChainedTransformation):
            yield from _flatten_transformations(t.transformations)
        else:
            yield t


def compress_transformations(
    transformations: Iterable[Transformation], /
) -> tuple[Transformation, ...]:
    """Compress any iterable of transformations by removing redundancies.

    Drops identity transformations and combines subsequent affine transformations.

    Args:
        transformations: An iterable of transformations.

    Returns:
        The minimum sequence of transformations that is equivalent to the input.
    """
    from baybe.transformations.basic import (
        AffineTransformation,
        BellTransformation,
        IdentityTransformation,
        TriangularTransformation,
        TwoSidedAffineTransformation,
    )

    aggregated: list[Transformation] = []
    id_ = IdentityTransformation()

    for t in _flatten_transformations(transformations):
        # Drop identity transformations (and such that are equivalent to it)
        if t == id_:
            continue

        last = aggregated.pop() if aggregated else None
        match (last, t):
            case AffineTransformation(), AffineTransformation():
                # Two subsequent affine transformations
                aggregated.append(combine_affine_transformations(last, t))
            case AffineTransformation(factor=1.0), BellTransformation():
                # Bell transformation after a pure input shift
                aggregated.append(evolve(t, center=t.center - last.shift))
            case AffineTransformation(factor=1.0), TwoSidedAffineTransformation():
                # 2-sided affine transformation after a pure input shift
                aggregated.append(evolve(t, midpoint=t.midpoint - last.shift))
            case AffineTransformation(factor=1.0), TriangularTransformation():
                # Triangular transformation after a pure input shift
                aggregated.append(
                    evolve(t, peak=t.peak - last.shift, cutoffs=t.cutoffs - last.shift)
                )
            case (None, _):
                aggregated.append(t)
            case (l, _):
                aggregated.append(l)
                aggregated.append(t)

    # Handle edge case when there was only a single identity transformation
    if not aggregated:
        return (IdentityTransformation(),)

    return tuple(aggregated)
