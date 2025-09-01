"""Transformation utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np

from baybe.transformations.base import Transformation

if TYPE_CHECKING:
    from torch import Tensor

    TensorCallable = Callable[[Tensor], Tensor]
    """Type alias for a torch-based function mapping from reals to reals."""


def convert_transformation(x: Transformation | TensorCallable, /) -> Transformation:
    """Autowrap a torch callable as transformation (with transformation passthrough)."""
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
    from baybe.transformations.basic import AffineTransformation, IdentityTransformation

    aggregated: list[Transformation] = []
    last = None
    id_ = IdentityTransformation()

    for t in _flatten_transformations(transformations):
        # Drop identity transformations (and such that are equivalent to it)
        if t == id_:
            continue

        # Combine subsequent affine transformations
        if (
            aggregated
            and isinstance(last := aggregated.pop(), AffineTransformation)
            and isinstance(t, AffineTransformation)
        ):
            aggregated.append(combine_affine_transformations(last, t))

        # Keep other transformations
        else:
            if last is not None:
                aggregated.append(last)
            aggregated.append(t)

    # Handle edge case when there was only a single identity transformation
    if not aggregated:
        return (IdentityTransformation(),)

    return tuple(aggregated)
