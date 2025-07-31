"""Composite transformations."""

from functools import reduce
from typing import Any

from attrs import define, field
from attrs.validators import and_, deep_iterable, instance_of, max_len, min_len
from torch import Tensor
from typing_extensions import override

from baybe.transformations.base import Transformation
from baybe.transformations.utils import compress_transformations
from baybe.utils.basic import compose
from baybe.utils.interval import Interval


@define
class ChainedTransformation(Transformation):
    """A chained transformation composing several individual transformations."""

    transformations: tuple[Transformation, ...] = field(
        converter=compress_transformations,
        validator=[
            min_len(1),
            deep_iterable(member_validator=instance_of(Transformation)),
        ],
    )
    """The transformations to be composed."""

    @override
    def __eq__(self, other: Any, /) -> bool:
        # A chained transformation with only one element is equivalent to that element
        if len(self.transformations) == 1:
            return self.transformations[0] == other

        # TODO: https://github.com/python-attrs/attrs/issues/1452
        return (
            super().__eq__(other)
            and isinstance(other, ChainedTransformation)
            and self.transformations == other.transformations
        )

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)
        return reduce(lambda acc, t: t.get_image(acc), self.transformations, interval)

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return compose(*(t.__call__ for t in self.transformations))(x)


@define
class AdditiveTransformation(Transformation):
    """A transformation implementing the sum of two transformations."""

    transformations: tuple[Transformation, Transformation] = field(
        converter=tuple,
        validator=deep_iterable(
            iterable_validator=and_(min_len(2), max_len(2)),
            member_validator=instance_of(Transformation),
        ),
    )
    """The transformations to be added."""

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        # NOTE: This only provides a conservative estimate of the image in that it
        #   computes an "upper bound" (i.e. an interval that contains the actual image),
        #   which is produced under the assumption that the extremal values of the two
        #   individual transformations occur at the same input value. Computing the
        #   exact image without additional information would require evaluating the
        #   transformation on the entire (uncountable) input space, which is infeasible.
        interval = Interval.create(interval)
        im1 = self.transformations[0].get_image(interval)
        im2 = self.transformations[1].get_image(interval)
        return Interval([im1.lower + im2.lower, im1.upper + im2.upper])

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return self.transformations[0](x) + self.transformations[1](x)


@define
class MultiplicativeTransformation(Transformation):
    """A transformation implementing the product of two transformations."""

    transformations: tuple[Transformation, Transformation] = field(
        converter=tuple,
        validator=deep_iterable(
            iterable_validator=and_(min_len(2), max_len(2)),
            member_validator=instance_of(Transformation),
        ),
    )
    """The transformations to be multiplied."""

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        # NOTE: This only provides a conservative estimate of the image in that it
        #   computes an "upper bound" (i.e. an interval that contains the actual image),
        #   which is produced under the assumption that the extremal values of the two
        #   individual transformations occur at the same input value. Computing the
        #   exact image without additional information would require evaluating the
        #   transformation on the entire (uncountable) input space, which is infeasible.
        interval = Interval.create(interval)
        im1 = self.transformations[0].get_image(interval)
        im2 = self.transformations[1].get_image(interval)
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
