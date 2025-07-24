"""Composite transformations."""

from functools import reduce
from typing import Any

from attrs import define, field
from attrs.validators import deep_iterable, instance_of, min_len
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
        return super().__eq__(other)

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)
        return reduce(lambda acc, t: t.get_image(acc), self.transformations, interval)

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return compose(*(t.__call__ for t in self.transformations))(x)
