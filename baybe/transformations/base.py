"""Base classes for target transformations."""

from __future__ import annotations

import gc
import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define, field, fields
from attrs.validators import deep_iterable, instance_of, min_len
from typing_extensions import override

from baybe.serialization.mixin import SerialMixin
from baybe.transformations.utils import compress_transformations
from baybe.utils.basic import is_all_instance
from baybe.utils.dataframe import to_tensor
from baybe.utils.interval import ConvertibleToInterval, Interval

if TYPE_CHECKING:
    from botorch.acquisition.objective import MCAcquisitionObjective
    from torch import Tensor


_TTransformation = TypeVar("_TTransformation", bound="Transformation")


def _image_equals_codomain(cls: type[_TTransformation], /) -> type[_TTransformation]:
    """Make the image of a transformation identical to its codomain."""
    cls.get_image = cls.get_codomain  # type: ignore[method-assign]
    return cls


@define(frozen=True)
class Transformation(SerialMixin, ABC):
    """Abstract base class for all transformations."""

    @abstractmethod
    def __call__(self, x: Tensor, /) -> Tensor:
        """Transform a given input tensor."""

    @abstractmethod
    def get_codomain(self, interval: Interval | None = None, /) -> Interval:
        """Get the codomain of a certain interval (assuming transformation continuity).

        In accordance with the mathematical definition of a function's `codomain
        <https://en.wikipedia.org/wiki/Codomain>`_, we define the codomain of a given
        :class:`~baybe.utils.interval.Interval` under a certain (assumed continuous)
        :class:`~Transformation` to be an :class:`~baybe.utils.interval.Interval`
        guaranteed to contain all possible outcomes when the :class:`~Transformation` is
        applied to all points in the input :class:`~baybe.utils.interval.Interval`. In
        cases where the image cannot exactly be computed, it is often still possible to
        compute a codomain. The codomain always contains the image, but might be larger.
        """

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
        # By default, it is assumed that the exact image of an interval cannot be
        # computed but only the codomain is available (see :meth:`get_codomain`).
        # Transformations that can provide the exact image should override this method.
        raise NotImplementedError(
            f"The exact image of the interval cannot be computed. "
            f"If sufficient, use '{self.get_codomain.__name__}' instead."
        )

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

    def hold_output_left_from(self, abscissa: float, /) -> Transformation:
        """Hold the output of the transformation left from a given abscissa value."""
        from baybe.transformations.basic import ClampingTransformation

        return ClampingTransformation(min=abscissa) | self

    def hold_output_right_from(self, abscissa: float, /) -> Transformation:
        """Hold the output of the transformation right from a given abscissa value."""
        from baybe.transformations.basic import ClampingTransformation

        return ClampingTransformation(max=abscissa) | self

    def hold_output_outside(self, interval: ConvertibleToInterval, /) -> Transformation:
        """Hold the output of the transformation outside a given interval."""
        from baybe.transformations.basic import ClampingTransformation

        return ClampingTransformation(*Interval.create(interval).to_tuple()) | self

    def abs(self) -> Transformation:
        """Take the absolute value of the output of the transformation."""
        from baybe.transformations.basic import AbsoluteTransformation

        return self | AbsoluteTransformation()

    def vshift(self, shift: float | int, /) -> Transformation:
        """Vertical shift (i.e. add a constant to the transformation output)."""
        return self + shift

    def hshift(self, shift: float | int, /) -> Transformation:
        """Horizontal shift (i.e. subtract a constant from the transformation input)."""
        from baybe.transformations import AffineTransformation

        return AffineTransformation(shift=-shift) | self

    def __neg__(self) -> Transformation:
        return self.negate()

    def __add__(self, other: Any) -> Transformation:
        """Add a constant or the output of another transformation."""
        if isinstance(other, Transformation):
            from baybe.transformations import AdditiveTransformation

            return AdditiveTransformation([self, other])
        if isinstance(other, (int, float)):
            from baybe.transformations import AffineTransformation

            return self | AffineTransformation(shift=other)
        return NotImplemented

    def __sub__(self, other: Any) -> Transformation:
        """Subtract a constant from the output of the transformation."""
        if isinstance(other, Transformation):
            from baybe.transformations import AdditiveTransformation

            return AdditiveTransformation([self, -other])
        if isinstance(other, (int, float)):
            from baybe.transformations import AffineTransformation

            return self | AffineTransformation(shift=-other)
        return NotImplemented

    def __mul__(self, other: Any) -> Transformation:
        """Multiply with a constant or the output of another transformation."""
        if isinstance(other, Transformation):
            from baybe.transformations import MultiplicativeTransformation

            return MultiplicativeTransformation([self, other])
        if isinstance(other, (int, float)):
            from baybe.transformations import AffineTransformation

            return self | AffineTransformation(factor=other)
        return NotImplemented

    def __truediv__(self, other: Any) -> Transformation:
        """Divide the output of the transformation by a constant."""
        if isinstance(other, (int, float)):
            from baybe.transformations import AffineTransformation

            if other == 0:
                raise ValueError("Division by zero is not allowed.")
            return self | AffineTransformation(factor=1 / other)
        return NotImplemented

    def __or__(self, other: Any) -> Transformation:
        """Chain the transformation with another one. Inspired by the Unix "pipe"."""
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
        if callable(other):
            from baybe.transformations.basic import CustomTransformation

            return self | CustomTransformation(other)
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


@_image_equals_codomain
@define(frozen=True)
class MonotonicTransformation(Transformation, ABC):
    """Abstract base class for monotonic transformations."""

    @override
    def get_codomain(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)
        return Interval(
            *sorted(
                [
                    self(to_tensor(interval.lower)).item(),
                    self(to_tensor(interval.upper)).item(),
                ]
            )
        )


@define(frozen=True)
class CompositeTransformation(Transformation, ABC):
    """Abstract base class for composite transformations."""

    transformations: tuple[Transformation, ...] = field(
        converter=compress_transformations,
        validator=[
            min_len(1),
            deep_iterable(member_validator=instance_of(Transformation)),
        ],
    )
    """The transformations to be composed."""

    @override
    def __new__(cls, *args: Any, **kwargs: Any) -> Transformation:  # type: ignore[misc]
        # If the transformations can be compressed into one, we return that instead
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


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
