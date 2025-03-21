"""Functions for bound transforms."""

from __future__ import annotations

import functools
from abc import ABC
from typing import Protocol, runtime_checkable
from collections.abc import Callable

import numpy as np
import torch
from attrs import define, field
from attrs.validators import deep_iterable, instance_of
from numpy.typing import ArrayLike
from torch import Tensor
from typing_extensions import override

from baybe.utils.basic import to_tuple

TensorCallable = Callable[[torch.Tensor], torch.Tensor]


def compose_two(f, g):
    return lambda *a, **kw: g(f(*a, **kw))


def compose(*fs):
    return functools.reduce(compose_two, fs)


@runtime_checkable
class TransformationProtocol(Protocol):
    def transform(self, x: Tensor, /) -> Tensor: ...


class Transformation(TransformationProtocol, ABC):
    def append(
        self, transformation: TransformationProtocol, /
    ) -> ChainedTransformation:
        return ChainedTransformation(self, transformation)

    def abs(self) -> Transformation:
        self.append(AbsoluteTransformation())


@define
class ChainedTransformation(Transformation):
    transformations: tuple[TransformationProtocol, ...] = field(
        converter=to_tuple,
        validator=deep_iterable(member_validator=instance_of(TransformationProtocol)),
    )

    def __init__(self, *transformations: TransformationProtocol):
        self.__attrs_init__(transformations)

    @override
    def append(
        self, transformation: TransformationProtocol, /
    ) -> ChainedTransformation:
        addendum = (
            transformation.transformations
            if isinstance(transformation, ChainedTransformation)
            else [transformation]
        )
        return ChainedTransformation(*self.transformations, *addendum)

    @override
    def transform(self, x: Tensor, /) -> Tensor:
        return compose(*(t.transform for t in self.transformations))(x)


@define
class GenericTransformation(Transformation):
    transformation: TensorCallable = field()

    @override
    def transform(self, x: Tensor, /) -> Tensor:
        return self.transformation(x)


@define
class ClampingTransformation(Transformation):
    min: float | None = field(default=None)
    max: float | None = field(default=None)

    @override
    def transform(self, x: Tensor, /) -> Tensor:
        return x.clamp(self.min, self.max)


@define(slots=False)
class AffineTransformation(Transformation):
    factor: float = field(default=1.0)
    shift: float = field(default=0.0)
    shift_first: bool = field(default=False)

    @classmethod
    def from_unit_interval(cls, lower: float, upper: float) -> AffineTransformation:
        return AffineTransformation(
            shift=-lower, factor=1 / (upper - lower), shift_first=True
        )

    @override
    def transform(self, x: Tensor, /) -> Tensor:
        if self.shift_first:
            return (x + self.shift) * self.factor
        else:
            return x * self.factor + self.shift


@define(slots=False)
class BellTransformation(Transformation):
    center: float = field(default=0.0)
    width: float = field(default=1.0)

    @override
    def transform(self, x: Tensor, /) -> Tensor:
        return x.sub(self.center).pow(2.0).div(2.0 * self.width**2).neg().exp()


class AbsoluteTransformation(Transformation):
    @override
    def transform(self, x: Tensor, /) -> Tensor:
        return x.abs()


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
