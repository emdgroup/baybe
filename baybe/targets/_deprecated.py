"""Numerical targets."""

from collections.abc import Callable, Sequence
from enum import Enum
from functools import partial

import numpy as np
from numpy.typing import ArrayLike


class TargetMode(Enum):
    """Available modes for targets."""

    MIN = "MIN"
    """The target is to be minimized."""

    MAX = "MAX"
    """The target is to be maximized."""

    MATCH = "MATCH"
    """The target should be close to a given value."""


class TargetTransformation(Enum):
    """Available target transformations."""

    LINEAR = "LINEAR"
    """Linear transformation."""

    TRIANGULAR = "TRIANGULAR"
    """Transformation using triangular-shaped function."""

    BELL = "BELL"
    """Transformation using bell-shaped function."""


_VALID_TRANSFORMATIONS: dict[TargetMode, Sequence[TargetTransformation]] = {
    TargetMode.MAX: (TargetTransformation.LINEAR,),
    TargetMode.MIN: (TargetTransformation.LINEAR,),
    TargetMode.MATCH: (TargetTransformation.TRIANGULAR, TargetTransformation.BELL),
}
"""A mapping from target modes to allowed target transformations.
If multiple transformations are allowed, the first entry is used as default option."""


def _get_target_transformation(
    mode: TargetMode, transformation: TargetTransformation
) -> Callable[[ArrayLike, float, float], np.ndarray]:
    """Provide the transform callable for the given target mode and transform type."""
    if transformation is TargetTransformation.TRIANGULAR:
        return triangular_transform
    if transformation is TargetTransformation.BELL:
        return bell_transform
    if transformation is TargetTransformation.LINEAR:
        if mode is TargetMode.MAX:
            return partial(linear_transform, descending=False)
        if mode is TargetMode.MIN:
            return partial(linear_transform, descending=True)
        raise ValueError(f"Unrecognized target mode: '{mode}'.")
    raise ValueError(f"Unrecognized target transformation: '{transformation}'.")


def linear_transform(
    arr: ArrayLike, lower: float, upper: float, descending: bool
) -> np.ndarray:
    """Linearly map values in a specified interval ``[lower, upper]`` to ``[0, 1]``.

    Outside the specified interval, the function remains constant.
    That is, 0 or 1, depending on the side and selected mode.

    Args:
        arr: The values to be mapped.
        lower: The lower boundary of the linear mapping interval.
        upper: The upper boundary of the linear mapping interval.
        descending: If ``True``, the function values decrease from 1 to 0 in the
            specified interval. If ``False``, they increase from 0 to 1.

    Returns:
        A new array containing the transformed values.
    """
    arr = np.asarray(arr)
    if descending:
        res = (upper - arr) / (upper - lower)
        res[arr > upper] = 0.0
        res[arr < lower] = 1.0
    else:
        res = (arr - lower) / (upper - lower)
        res[arr > upper] = 1.0
        res[arr < lower] = 0.0

    return res


def triangular_transform(arr: ArrayLike, lower: float, upper: float) -> np.ndarray:
    """Map values to the interval ``[0, 1]`` in a "triangular" fashion.

    The shape of the function is "triangular" in that is 0 outside a specified interval
    and linearly increases to 1 from both interval ends, reaching the value 1 at the
    center of the interval.

    Args:
        arr: The values to be mapped.
        lower: The lower end of the triangle interval. Below, the mapped values are 0.
        upper:The upper end of the triangle interval. Above, the mapped values are 0.

    Returns:
        A new array containing the transformed values.
    """
    arr = np.asarray(arr)
    mid = lower + (upper - lower) / 2
    res = (arr - lower) / (mid - lower)
    res[arr > mid] = (upper - arr[arr > mid]) / (upper - mid)
    res[arr > upper] = 0.0
    res[arr < lower] = 0.0

    return res


def bell_transform(arr: ArrayLike, lower: float, upper: float) -> np.ndarray:
    """Map values to the interval ``[0, 1]`` in a "Gaussian bell" fashion.

    The shape of the function is "Gaussian bell curve", specified through the boundary
    values of the sigma interval. Reaches the maximum value of 1 at the interval center.

    Args:
        arr: The values to be mapped.
        lower: The input value corresponding to the upper sigma interval boundary.
        upper: The input value corresponding to the lower sigma interval boundary.

    Returns:
        A new array containing the transformed values.
    """
    arr = np.asarray(arr)
    mean = np.mean([lower, upper])
    std = (upper - lower) / 2
    res = np.exp(-((arr - mean) ** 2) / (2.0 * std**2))

    return res
