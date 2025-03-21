"""Functions for bound transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from attrs import define, field
from attrs.converters import optional
from attrs.validators import deep_iterable, instance_of, is_callable
from typing_extensions import override

from baybe.targets._deprecated import (  # noqa: F401
    bell_transform,
    linear_transform,
    triangular_transform,
)
from baybe.utils.basic import compose, to_tuple

if TYPE_CHECKING:
    from torch import Tensor

    TensorCallable = Callable[[Tensor], Tensor]
    """Type alias for a torch-based function mapping from reals to reals."""


def convert_transformation(
    x: TransformationProtocol | TensorCallable, /
) -> TransformationProtocol:
    """Autowrap a torch callable as transformation (with transformation passthrough)."""
    return x if isinstance(x, TransformationProtocol) else GenericTransformation(x)


@runtime_checkable
class TransformationProtocol(Protocol):
    """Type protocol specifying the interface transformations need to implement."""

    def transform(self, x: Tensor, /) -> Tensor:
        """Transform a given input tensor."""


@define
class Transformation(TransformationProtocol, ABC):
    """Abstract base class for all transformations."""

    @override
    @abstractmethod
    def transform(self, x: Tensor, /) -> Tensor:
        """Transform a given input tensor."""

    def append(
        self, transformation: TransformationProtocol, /
    ) -> ChainedTransformation:
        """Chain another transformation with the existing one."""
        return self + transformation

    def negate(self) -> Transformation:
        """Negate the output of the transformation."""
        return self + AffineTransformation(factor=-1)

    def clamp(self, min: float | None, max: float | None) -> Transformation:
        """Clamp the output of the transformation."""
        return self + ClampingTransformation(min, max)

    def abs(self) -> Transformation:
        """Take the absolute value of the output of the transformation."""
        return self + AbsoluteTransformation()

    def __add__(
        self, other: TransformationProtocol | int | float
    ) -> ChainedTransformation:
        """Chain another transformation or shift the output of the current one."""
        if isinstance(other, TransformationProtocol):
            return ChainedTransformation(self, other)
        if isinstance(other, (int, float)):
            return self + AffineTransformation(shift=other)
        return NotImplemented

    def __mul__(self, other: TransformationProtocol) -> ChainedTransformation:
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


@define(init=False)
class ChainedTransformation(Transformation):
    """A chained transformation composing several individual transformations."""

    transformations: tuple[TransformationProtocol, ...] = field(
        converter=to_tuple,
        validator=deep_iterable(member_validator=instance_of(TransformationProtocol)),
    )
    """The transformations to be composed."""

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
    """A generic transformation applying an arbitrary torch callable."""

    transformation: TensorCallable = field(validator=is_callable())
    """The torch callable to be applied."""

    @override
    def transform(self, x: Tensor, /) -> Tensor:
        return self.transformation(x)


@define
class ClampingTransformation(Transformation):
    """A transformation clamping values between specified cutoffs."""

    min: float | None = field(default=None, converter=optional(float))
    """The lower cutoff value."""

    max: float | None = field(default=None, converter=optional(float))
    """The upper cutoff value."""

    @override
    def transform(self, x: Tensor, /) -> Tensor:
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
            >>> t = AffineTransformation.from_unit_interval(3, 7)
            >>> t.transform(torch.tensor([3, 7]))
            tensor([0., 1.])
            >>> t.transform(torch.tensor([7, 3]))
            tensor([1., 0.])
        """
        return AffineTransformation(
            shift=-mapped_to_zero,
            factor=1 / (mapped_to_one - mapped_to_zero),
            shift_first=True,
        )

    @override
    def transform(self, x: Tensor, /) -> Tensor:
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
    def transform(self, x: Tensor, /) -> Tensor:
        return x.sub(self.center).pow(2.0).div(2.0 * self.width**2).neg().exp()


class AbsoluteTransformation(Transformation):
    """A transformation computing absolute values."""

    @override
    def transform(self, x: Tensor, /) -> Tensor:
        return x.abs()
