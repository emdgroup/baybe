"""Composite transformations."""

from __future__ import annotations

import gc
from functools import reduce
from typing import TYPE_CHECKING, Any

from attrs import define, field
from attrs.validators import and_, deep_iterable, instance_of, max_len, min_len
from typing_extensions import override

from baybe.transformations.base import Transformation
from baybe.transformations.utils import compress_transformations
from baybe.utils.basic import compose, to_tuple
from baybe.utils.interval import Interval

if TYPE_CHECKING:
    from torch import Tensor


@define(frozen=True)
class ChainedTransformation(Transformation):
    """A chained transformation composing several individual transformations."""

    # https://github.com/python-attrs/attrs/issues/1462
    __hash__ = object.__hash__

    transformations: tuple[Transformation, ...] = field(
        converter=compress_transformations,
        validator=[
            min_len(1),
            deep_iterable(member_validator=instance_of(Transformation)),
        ],
    )
    """The transformations to be composed (the first element gets applied first)."""

    @override
    def __eq__(self, other: Any, /) -> bool:
        if len(self.transformations) == 1:
            # A chained transformation with only one element is equivalent to that
            # element
            return self.transformations[0] == other
        if isinstance(other, ChainedTransformation):
            return self.transformations == other.transformations
        return NotImplemented

    @override
    def get_codomain(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)
        return reduce(
            lambda acc, t: t.get_codomain(acc), self.transformations, interval
        )

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)
        return reduce(lambda acc, t: t.get_image(acc), self.transformations, interval)

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return compose(*(t.__call__ for t in self.transformations))(x)


@define(frozen=True)
class AdditiveTransformation(Transformation):
    """A transformation implementing the sum of two transformations."""

    transformations: tuple[Transformation, Transformation] = field(
        converter=to_tuple,
        validator=deep_iterable(
            iterable_validator=and_(min_len(2), max_len(2)),
            member_validator=instance_of(Transformation),
        ),
    )
    """The transformations to be added."""

    @override
    def get_codomain(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)
        im1 = self.transformations[0].get_codomain(interval)
        im2 = self.transformations[1].get_codomain(interval)
        return Interval(im1.lower + im2.lower, im1.upper + im2.upper)

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return self.transformations[0](x) + self.transformations[1](x)


@define(frozen=True)
class MultiplicativeTransformation(Transformation):
    """A transformation implementing the product of two transformations."""

    transformations: tuple[Transformation, Transformation] = field(
        converter=to_tuple,
        validator=deep_iterable(
            iterable_validator=and_(min_len(2), max_len(2)),
            member_validator=instance_of(Transformation),
        ),
    )
    """The transformations to be multiplied."""

    @override
    def get_codomain(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)
        im1 = self.transformations[0].get_codomain(interval)
        im2 = self.transformations[1].get_codomain(interval)
        boundary_products = [
            im1.lower * im2.lower,
            im1.lower * im2.upper,
            im1.upper * im2.lower,
            im1.upper * im2.upper,
        ]
        return Interval(min(boundary_products), max(boundary_products))

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return self.transformations[0](x) * self.transformations[1](x)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
