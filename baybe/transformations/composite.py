"""Composite transformations."""

from __future__ import annotations

import gc
from functools import reduce
from typing import TYPE_CHECKING

from attrs import define
from typing_extensions import override

from baybe.transformations.base import CompositeTransformation
from baybe.utils.basic import compose
from baybe.utils.interval import Interval

if TYPE_CHECKING:
    from torch import Tensor


@define(frozen=True)
class ChainedTransformation(CompositeTransformation):
    """A chained transformation composing several individual transformations.

    The first transformations in the chain is applied first.
    """

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
class AdditiveTransformation(CompositeTransformation):
    """A transformation implementing the sum of two transformations."""

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
class MultiplicativeTransformation(CompositeTransformation):
    """A transformation implementing the product of two transformations."""

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
