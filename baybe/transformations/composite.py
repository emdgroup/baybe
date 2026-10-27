"""Composite transformations."""

from __future__ import annotations

import gc
import inspect
from functools import reduce
from typing import TYPE_CHECKING, Any

from attrs import define, field, fields
from attrs.validators import deep_iterable, instance_of, max_len, min_len
from typing_extensions import override

from baybe.transformations.base import Transformation
from baybe.transformations.utils import (
    compress_transformations,
    sum_affine_transformations,
)
from baybe.utils.basic import compose, to_tuple
from baybe.utils.interval import Interval

if TYPE_CHECKING:
    from torch import Tensor


@define(frozen=True)
class ChainedTransformation(Transformation):
    r"""Apply a sequence of transformations in the listed order.

    For ``transformations = (t1, t2, ..., tn)``, computes
    :math:`f(x) = t_n(\cdots t_2(t_1(x)) \cdots)`. Each receives the previous
    one's output, so order matters. ``t1 | t2`` is equivalent syntax for two steps.
    """

    transformations: tuple[Transformation, ...] = field(
        converter=compress_transformations,
        validator=[
            min_len(1),
            deep_iterable(member_validator=instance_of(Transformation)),
        ],
    )
    """The transformations in application order, first element first."""

    @override
    def __new__(cls, *args: Any, **kwargs: Any) -> Transformation:  # type: ignore[misc]
        # If the transformations can be condensed into one, we return that instead
        sig = inspect.signature(cls.__init__)
        try:
            bound = sig.bind(None, *args, **kwargs)
        except TypeError:
            # Unpickling first creates an uninitialized instance, i.e. the args/kwargs
            # are not provided. Hence, we simply return an raw object.
            return super().__new__(cls)
        transformations = bound.arguments[fields(cls).transformations.name]
        if len(compressed := compress_transformations(transformations)) == 1:
            return compressed[0]
        return super().__new__(cls)

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
    """Sum the outputs of two transformations applied to the same input.

    For ``transformations = (t1, t2)``, computes :math:`f(x) = t_1(x) + t_2(x)`.
    Both receive :math:`x` independently, not each other's output. ``t1 + t2`` is
    equivalent syntax; more terms can be combined by nesting additions.
    """

    transformations: tuple[Transformation, Transformation] = field(
        converter=to_tuple,
        validator=deep_iterable(
            iterable_validator=(min_len(2), max_len(2)),
            member_validator=instance_of(Transformation),
        ),
    )
    """The two transformations whose outputs for the same input are added."""

    @override
    def __new__(cls, *args: Any, **kwargs: Any) -> Transformation:  # type: ignore[misc]
        # If the transformations can be condensed into one, we return that instead
        sig = inspect.signature(cls.__init__)
        try:
            bound = sig.bind(None, *args, **kwargs)
        except TypeError:
            # Unpickling first creates an uninitialized instance, i.e. the args/kwargs
            # are not provided. Hence, we simply return an raw object.
            return super().__new__(cls)
        transformations = bound.arguments[fields(cls).transformations.name]
        if len(compressed := sum_affine_transformations(transformations)) == 1:
            return compressed.pop()
        return super().__new__(cls)

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
    r"""Multiply the outputs of two transformations applied to the same input.

    For ``transformations = (t1, t2)``, computes :math:`f(x) = t_1(x) \cdot t_2(x)`.
    Both receive :math:`x` independently, not each other's output. ``t1 * t2`` is
    equivalent syntax; more factors can be combined by nesting multiplications.
    """

    transformations: tuple[Transformation, Transformation] = field(
        converter=to_tuple,
        validator=deep_iterable(
            iterable_validator=(min_len(2), max_len(2)),
            member_validator=instance_of(Transformation),
        ),
    )
    """The two transformations whose outputs for the same input are multiplied."""

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
