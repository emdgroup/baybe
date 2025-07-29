"""Base classes for target transformations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from attrs import define
from typing_extensions import override

from baybe.serialization.mixin import SerialMixin
from baybe.utils.basic import is_all_instance
from baybe.utils.dataframe import to_tensor
from baybe.utils.interval import Interval

if TYPE_CHECKING:
    from botorch.acquisition.objective import MCAcquisitionObjective
    from torch import Tensor


@define
class Transformation(SerialMixin, ABC):
    """Abstract base class for all transformations."""

    @abstractmethod
    def __call__(self, x: Tensor, /) -> Tensor:
        """Transform a given input tensor."""

    @abstractmethod
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        """Get the image of a certain interval (assuming transformation continuity).

        In accordance with the mathematical definition of a function's `image
        <https://en.wikipedia.org/wiki/Image_(mathematics)>`_, we define the image of a
        given :class:`~baybe.utils.interval.Interval` under a certain (assumed
        continuous) :class:`~Transformation` to be the smallest
        :class:`~baybe.utils.interval.Interval` containing all possible outcomes when
        the :class:`~Transformation` is applied to all points in the input
        :class:`~baybe.utils.interval.Interval`.
        """

    def to_botorch_objective(self) -> MCAcquisitionObjective:
        """Convert to BoTorch objective."""
        from botorch.acquisition.objective import GenericMCObjective

        return GenericMCObjective(lambda samples, X: self(samples))

    def chain(self, transformation: Transformation, /) -> Transformation:
        """Chain another transformation with the existing one."""
        return self | transformation

    def negate(self) -> Transformation:
        """Negate the output of the transformation."""
        from baybe.transformations.basic import AffineTransformation

        return self | AffineTransformation(factor=-1)

    def clamp(
        self, min: float = float("-inf"), max: float = float("inf")
    ) -> Transformation:
        """Clamp the output of the transformation."""
        if min == float("-inf") and max == float("inf"):
            raise ValueError(
                "A clamping transformation requires at least one finite boundary value."
            )
        from baybe.transformations.basic import ClampingTransformation

        return self | ClampingTransformation(min, max)

    def abs(self) -> Transformation:
        """Take the absolute value of the output of the transformation."""
        from baybe.transformations.basic import AbsoluteTransformation

        return self | AbsoluteTransformation()

    def __add__(self, other: Transformation | int | float) -> Transformation:
        """Add a constant or the output from another transformation."""
        if isinstance(other, Transformation):
            from baybe.transformations import AdditiveTransformation

            return AdditiveTransformation([self, other])
        if isinstance(other, (int, float)):
            from baybe.transformations import AffineTransformation

            return self | AffineTransformation(shift=other)
        return NotImplemented

    def __mul__(self, other: Transformation | int | float) -> Transformation:
        """Multiply with a constant or the output from another transformation."""
        if isinstance(other, Transformation):
            from baybe.transformations import MultiplicativeTransformation

            return MultiplicativeTransformation([self, other])
        if isinstance(other, (int, float)):
            from baybe.transformations import AffineTransformation

            return self | AffineTransformation(shift=other)
        return NotImplemented

    def __or__(self, other: Transformation) -> Transformation:
        """Chain the transformation with another one."""
        from baybe.transformations import (
            AffineTransformation,
            ChainedTransformation,
            IdentityTransformation,
            combine_affine_transformations,
        )

        if isinstance(other, IdentityTransformation):
            return self
        if is_all_instance(t := [self, other], AffineTransformation):
            return combine_affine_transformations(*t)
        if isinstance(other, Transformation):
            return ChainedTransformation([self, other])
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

        from baybe.transformations.basic import CustomTransformation

        return args[0] | CustomTransformation(func)


class MonotonicTransformation(Transformation, ABC):
    """Abstract base class for monotonic transformations."""

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)
        return Interval(
            *sorted(
                [
                    self(to_tensor(interval.lower)).item(),
                    self(to_tensor(interval.upper)).item(),
                ]
            )
        )
