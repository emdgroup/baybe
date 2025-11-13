"""Transformation utilities."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from typing import TYPE_CHECKING

import numpy as np
from attrs import evolve

from baybe.transformations.basic import AffineTransformation, IdentityTransformation
from baybe.utils.basic import is_all_instance

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
    """Compress any chain of transformations by removing redundancies.

    Drops identity transformations and combines subsequent affine transformations.

    Args:
        transformations: An iterable of transformations.

    Raises:
        TypeError: If any of the passed elements is not a
            :class:`baybe.transformations.base.Transformation`.

    Returns:
        The minimum sequence of transformations that is equivalent to the input.
    """
    from baybe.transformations.base import Transformation
    from baybe.transformations.basic import (
        AffineTransformation,
        BellTransformation,
        IdentityTransformation,
        TriangularTransformation,
        TwoSidedAffineTransformation,
    )

    transformations = list(transformations)
    if not is_all_instance(transformations, Transformation):
        raise TypeError(f"All elements must be of type '{Transformation.__name__}'.")

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


def merge_affine_transformations(
    transformations: Collection[Transformation], /
) -> set[Transformation]:
    """Additively combine affine transformations in a given transformation collection.

    Args:
        transformations: A collection of transformations.

    Raises:
        TypeError: If any of the passed elements is not a
            :class:`baybe.transformations.base.Transformation`.

    Returns:
        A condensed version of the collection where all affine transformations have been
        combined into one.
    """
    from baybe.transformations.base import Transformation

    if not is_all_instance(transformations, Transformation):
        raise TypeError(f"All elements must be of type '{Transformation.__name__}'.")

    # Split into affine and non-affine transformations
    is_affine = [
        isinstance(tr, (IdentityTransformation, AffineTransformation))
        for tr in transformations
    ]
    affines = [tr for tr, is_aff in zip(transformations, is_affine) if is_aff]
    non_affines = [tr for tr, is_aff in zip(transformations, is_affine) if not is_aff]

    # Compute the combined affine transformation coeffiecients
    factor = sum(
        tr.factor if isinstance(tr, AffineTransformation) else 1.0 for tr in affines
    )
    shift = sum(
        tr.shift if isinstance(tr, AffineTransformation) else 0.0 for tr in affines
    )

    return {AffineTransformation(factor=factor, shift=shift), *non_affines}
