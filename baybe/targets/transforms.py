"""Functions for bound transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
from attrs import define, field
from attrs.validators import deep_iterable, instance_of, is_callable, min_len
from typing_extensions import override

from baybe.targets._deprecated import (  # noqa: F401
    bell_transform,
    linear_transform,
    triangular_transform,
)
from baybe.utils.basic import compose, to_tuple
from baybe.utils.interval import Interval

if TYPE_CHECKING:
    from botorch.acquisition.objective import MCAcquisitionObjective
    from torch import Tensor

    TensorCallable = Callable[[Tensor], Tensor]
    """Type alias for a torch-based function mapping from reals to reals."""


def convert_transformation(x: Transformation | TensorCallable, /) -> Transformation:
    """Autowrap a torch callable as transformation (with transformation passthrough)."""
    return x if isinstance(x, Transformation) else GenericTransformation(x)


@define
class Transformation(ABC):
    """Abstract base class for all transformations."""

    @abstractmethod
    def __call__(self, x: Tensor, /) -> Tensor:
        """Transform a given input tensor."""

    @abstractmethod
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        """Get the image of a certain interval (assuming transformation continuity)."""

    def to_botorch(self, *, keep_dimension: bool) -> MCAcquisitionObjective:
        """Convert to BoTorch representation.

        Args:
            keep_dimension: Boolean flag specifying whether to keep an explicit
                dimension for the outputs of the objective.

        Returns:
            The BoTorch objective.
        """
        from botorch.acquisition.objective import GenericMCObjective

        if keep_dimension:
            return GenericMCObjective(lambda samples, X: self(samples))
        else:
            return GenericMCObjective(lambda samples, X: self(samples).squeeze(-1))

    def append(self, transformation: Transformation, /) -> Transformation:
        """Chain another transformation with the existing one."""
        return self + transformation

    def negate(self) -> Transformation:
        """Negate the output of the transformation."""
        return self + AffineTransformation(factor=-1)

    def clamp(
        self, min: float = float("-inf"), max: float = float("inf")
    ) -> Transformation:
        """Clamp the output of the transformation."""
        if min == float("-inf") and max == float("inf"):
            raise ValueError(
                "A clamping transformation requires at least one finite boundary value."
            )

        return self + ClampingTransformation(min, max)

    def abs(self) -> Transformation:
        """Take the absolute value of the output of the transformation."""
        return self + AbsoluteTransformation()

    def __add__(self, other: Transformation | int | float) -> Transformation:
        """Chain another transformation or shift the output of the current one."""
        if isinstance(other, IdentityTransformation):
            return self
        if isinstance(other, Transformation):
            return ChainedTransformation([self, other])
        if isinstance(other, (int, float)):
            return self + AffineTransformation(shift=other)
        return NotImplemented

    def __mul__(self, other: Transformation) -> ChainedTransformation:
        """Scale the output of the transformation."""
        if isinstance(other, (int, float)):
            return self + AffineTransformation(factor=other)
        return NotImplemented

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Chain the transformation with a given torch callable."""
        if not (
            len(args) == 1 and isinstance(args[0], Transformation) and kwargs is None
        ):
            raise ValueError(
                "Composing transformations with torch operations is only supported "
                "if the transformation enters as the only (positional) argument."
            )
        return args[0] + GenericTransformation(func)


@define
class ChainedTransformation(Transformation):
    """A chained transformation composing several individual transformations."""

    transformations: tuple[Transformation, ...] = field(
        converter=lambda x: to_tuple(
            t for t in x if not isinstance(t, (IdentityTransformation, type(None)))
        ),
        validator=[
            min_len(2),
            deep_iterable(member_validator=instance_of(Transformation)),
        ],
    )
    """The transformations to be composed."""

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)
        return reduce(lambda acc, t: t.get_image(acc), self.transformations, interval)

    @override
    def append(self, transformation: Transformation, /) -> ChainedTransformation:
        addendum = (
            transformation.transformations
            if isinstance(transformation, ChainedTransformation)
            else [transformation]
        )
        return ChainedTransformation([*self.transformations, *addendum])

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return compose(*(t.__call__ for t in self.transformations))(x)


@define
class GenericTransformation(Transformation):
    """A generic transformation applying an arbitrary torch callable."""

    transformation: TensorCallable = field(validator=is_callable())
    """The torch callable to be applied."""

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        raise NotImplementedError(
            "Generic transformations do not provide details about their image."
        )

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return self.transformation(x)


@define
class IdentityTransformation(Transformation):
    """The identity transformation."""

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        return Interval.create(interval)

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return x

    @override
    def __add__(self, other: Transformation | int | float) -> Transformation:
        if isinstance(other, (int, float)):
            return AffineTransformation(shift=other)
        return other


@define
class ClampingTransformation(Transformation):
    """A transformation clamping values between specified cutoffs."""

    min: float = field(default=float("-inf"), converter=float)
    """The lower cutoff value."""

    max: float = field(default=float("inf"), converter=float)
    """The upper cutoff value."""

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)
        return Interval(
            max(min(interval.lower, self.max), self.min),
            min(max(interval.upper, self.min), self.max),
        )

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return x.clamp(self.min, self.max)


@define(slots=False)
class AffineTransformation(Transformation):
    """An affine transformation."""

    factor: float = field(default=1.0, converter=float)
    """The multiplicative factor of the transformation."""

    shift: float = field(default=0.0, converter=float)
    """The constant shift of the transformation."""

    shift_first: bool = field(default=False, validator=instance_of(bool))
    """Boolean flag determining if the shift or the scaling is applied first."""

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)

        import torch

        return Interval(
            *sorted(
                [
                    float(self(torch.tensor(interval.lower))),
                    float(self(torch.tensor(interval.upper))),
                ]
            )
        )

    @classmethod
    def from_unit_interval(
        cls, mapped_to_zero: float, mapped_to_one: float
    ) -> AffineTransformation:
        """Create an affine transform by specifying reference points mapped to 0/1.

        Args:
            mapped_to_zero: The input value that will be mapped to zero.
            mapped_to_one: The input value that will be mapped to one.

        Returns:
            An affine transformation calibrated to map the specified values to the
            unit interval.

        Example:
            >>> import torch
            >>> from baybe.targets.transforms import AffineTransformation
            >>> transform = AffineTransformation.from_unit_interval(3, 7)
            >>> transform(torch.tensor([3, 7]))
            tensor([0., 1.])
            >>> transform(torch.tensor([7, 3]))
            tensor([1., 0.])
        """
        return AffineTransformation(
            shift=-mapped_to_zero,
            factor=1 / (mapped_to_one - mapped_to_zero),
            shift_first=True,
        )

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        if self.shift_first:
            return (x + self.shift) * self.factor
        else:
            return x * self.factor + self.shift


@define(slots=False)
class BellTransformation(Transformation):
    """A Gaussian bell curve transformation."""

    center: float = field(default=0.0, converter=float)
    """The center point of the bell curve."""

    width: float = field(default=1.0, converter=float)
    """The width of the bell curve."""

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)

        import torch

        image_lower = float(self(torch.tensor(interval.lower)))
        image_upper = float(self(torch.tensor(interval.upper)))
        if interval.contains(self.center):
            return Interval(min(image_lower, image_upper), 1)
        else:
            return Interval(*sorted([image_lower, image_upper]))

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return x.sub(self.center).pow(2.0).div(2.0 * self.width**2).neg().exp()


@define(slots=False)
class AbsoluteTransformation(Transformation):
    """A transformation computing absolute values."""

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)

        image_lower = abs(interval.lower)
        image_upper = abs(interval.upper)
        if interval.contains(0):
            return Interval(0, max(image_lower, image_upper))
        else:
            return Interval(*sorted([image_lower, image_upper]))

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return x.abs()


@define(frozen=True)
class LogarithmicTransformation(Transformation):
    """A logarithmic transformation."""

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)
        return Interval(np.log(interval.lower), np.log(interval.upper))

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return x.log()


@define(frozen=True)
class ExponentialTransformation(Transformation):
    """An exponential transformation."""

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)
        return Interval(np.exp(interval.lower), np.exp(interval.upper))

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return x.exp()


@define(slots=False)
class PowerTransformation(Transformation):
    """A transformation computing the power."""

    exponent: float = field(converter=float)
    """The exponent of the power transformation."""

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)
        return Interval(interval.lower**self.exponent, interval.upper**self.exponent)

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return x.pow(self.exponent)
