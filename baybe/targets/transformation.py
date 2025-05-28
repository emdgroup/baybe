"""Target transformations."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from functools import reduce
from typing import TYPE_CHECKING

from attrs import define, field
from attrs.validators import deep_iterable, instance_of, is_callable, min_len
from typing_extensions import override

from baybe.serialization.core import (
    converter,
    get_base_structure_hook,
    unstructure_base,
)
from baybe.serialization.mixin import SerialMixin
from baybe.targets._deprecated import (  # noqa: F401
    bell_transform,
    linear_transform,
    triangular_transform,
)
from baybe.utils.basic import compose, is_all_instance
from baybe.utils.dataframe import to_tensor
from baybe.utils.interval import Interval

if TYPE_CHECKING:
    from botorch.acquisition.objective import MCAcquisitionObjective
    from torch import Tensor

    from baybe.targets.botorch import AffinePosteriorTransform

    TensorCallable = Callable[[Tensor], Tensor]
    """Type alias for a torch-based function mapping from reals to reals."""


def convert_transformation(x: Transformation | TensorCallable, /) -> Transformation:
    """Autowrap a torch callable as transformation (with transformation passthrough)."""
    return x if isinstance(x, Transformation) else GenericTransformation(x)


@define
class Transformation(SerialMixin, ABC):
    """Abstract base class for all transformations."""

    @abstractmethod
    def __call__(self, x: Tensor, /) -> Tensor:
        """Transform a given input tensor."""

    @abstractmethod
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        """Get the image of a certain interval (assuming transformation continuity)."""

    def to_botorch_objective(self) -> MCAcquisitionObjective:
        """Convert to BoTorch objective."""
        from botorch.acquisition.objective import GenericMCObjective

        return GenericMCObjective(lambda samples, X: self(samples))

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
        if is_all_instance(t := [self, other], AffineTransformation):
            return combine_affine_transformations(*t)
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


class MonotonicTransformation(Transformation):
    """Class for monotonic transformations."""

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


def combine_affine_transformations(
    t1: AffineTransformation, t2: AffineTransformation, /
) -> AffineTransformation:
    """Combine two affine transformations into one."""
    return AffineTransformation(
        factor=t2.factor * t1.factor,
        shift=t2.factor * t1.shift + t2.shift,
    )


def _flatten_transformations(
    transformations: Iterable[Transformation], /
) -> Iterable[Transformation]:
    """Recursively flatten nested chained transformations."""
    for t in transformations:
        if isinstance(t, ChainedTransformation):
            yield from _flatten_transformations(t.transformations)
        else:
            yield t


def compress_transformations(
    transformations: Iterable[Transformation], /
) -> tuple[Transformation, ...]:
    """Compress any iterable of transformations by removing redundancies.

    Drops identity transformations and combines subsequent affine transformations.

    Args:
        transformations: An iterable of transformations.

    Returns:
        The minimum sequence of transformations that is equivalent to the input.
    """
    aggregated: list[Transformation] = []
    last = None

    for t in _flatten_transformations(transformations):
        # Drop identity transformations
        if isinstance(t, IdentityTransformation):
            continue

        # Combine subsequent affine transformations
        if (
            aggregated
            and isinstance(last := aggregated.pop(), AffineTransformation)
            and isinstance(t, AffineTransformation)
        ):
            aggregated.append(combine_affine_transformations(last, t))

        # Keep other transformations
        else:
            if last is not None:
                aggregated.append(last)
            aggregated.append(t)

    return tuple(aggregated)


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
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)
        return reduce(lambda acc, t: t.get_image(acc), self.transformations, interval)

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
class IdentityTransformation(MonotonicTransformation):
    """The identity transformation."""

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return x

    @override
    def __add__(self, other: Transformation | int | float) -> Transformation:
        if isinstance(other, (int, float)):
            return AffineTransformation(shift=other)
        return other


@define
class ClampingTransformation(MonotonicTransformation):
    """A transformation clamping values between specified cutoffs."""

    min: float = field(default=float("-inf"), converter=float)
    """The lower cutoff value."""

    max: float = field(default=float("inf"), converter=float)
    """The upper cutoff value."""

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return x.clamp(self.min, self.max)


@define(slots=False, init=False)
class AffineTransformation(MonotonicTransformation):
    """An affine transformation."""

    factor: float = field(default=1.0, converter=float)
    """The multiplicative factor of the transformation."""

    shift: float = field(default=0.0, converter=float)
    """The constant shift of the transformation."""

    def __init__(
        self,
        factor: float = 1.0,
        shift: float = 0.0,
        shift_first: bool = False,
    ) -> None:
        shift = shift * factor if shift_first else shift
        self.__attrs_init__(factor=factor, shift=shift)

    def to_botorch_posterior_transform(self) -> AffinePosteriorTransform:
        """Convert to BoTorch posterior transform.

        Returns:
            The representation of the transform as BoTorch posterior transform.
        """
        from baybe.targets.botorch import AffinePosteriorTransform

        return AffinePosteriorTransform(self.factor, self.shift)

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
            >>> from baybe.targets.transformation import AffineTransformation
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
        return x * self.factor + self.shift


@define(slots=False)
class TwoSidedLinearTransformation(Transformation):
    """A transformation with two linear segments on either side of a center point."""

    slope_left: float = field(converter=float)
    """The slope of the linear segment to the left of the center."""

    slope_right: float = field(converter=float)
    """The slope of the linear segment to the right of the center."""

    center: float = field(default=0.0, converter=float)
    """The center point of the transformation."""

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        interval = Interval.create(interval)

        image_lower = self(to_tensor(interval.lower)).item()
        image_upper = self(to_tensor(interval.upper)).item()
        min_val, max_val = sorted([image_lower, image_upper])
        if interval.contains(self.center):
            return Interval(min(0, min_val), max(0, max_val))
        else:
            return Interval(min_val, max_val)

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        import torch

        return torch.where(
            x < self.center,
            (x - self.center) * self.slope_left,
            (x - self.center) * self.slope_right,
        )


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

        image_lower = self(to_tensor(interval.lower)).item()
        image_upper = self(to_tensor(interval.upper)).item()
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

    _transformation: Transformation = field(
        factory=lambda: TwoSidedLinearTransformation(slope_left=-1, slope_right=1),
        init=False,
        repr=False,
    )
    """Internal transformation object handling the operations."""

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        return self._transformation.get_image(interval)

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return self._transformation(x)


@define(slots=False)
class TriangularTransformation(Transformation):
    r"""A transformation with a triangular shape.

    The transformation is defined by a peak location between two cutoff values. Outside
    the region delimited by the cutoff values, the transformation is zero. Inside the
    region, the transformed values increase linearly from both cutoffs to the peak,
    where the highest value of 1 is reached:

    .. math::
        f(x) =
        \begin{cases}
            0 & \text{if } x < c_1 \\
            \frac{x - c_1}{p - c_1} & \text{if } c_1 \leq x < p \\
            \frac{c_2 - x}{c_2 - p} & \text{if } p \leq x < c_2 \\
            0 & \text{if } c_2 \leq x
        \end{cases}

    where :math:`c_1` and :math:`c_2` are the left and right cutoffs, respectively, and
    :math:`p` is the peak location, with :math:`c_1 < p < c_2`.
    """

    peak: float = field(converter=float)
    """The location of the peak of the transformation."""

    cutoffs: Interval = field(converter=Interval.create)
    """The cutoff values where the transformation reaches zero."""

    _transformation: Transformation = field(init=False, repr=False)
    """Internal transformation object handling the operations."""

    @_transformation.default
    def _default_transformation(self) -> Transformation:
        return (
            TwoSidedLinearTransformation(
                slope_left=1 / self.margins[0],
                slope_right=-1 / self.margins[1],
                center=self.peak,
            )
            + 1
        ).clamp(min=0)

    @cutoffs.validator
    def _validate_cutoffs(self, _, cutoffs: Interval) -> None:
        if not (cutoffs.lower < self.peak < cutoffs.upper):
            raise ValueError(
                f"The peak of the transformation must be located strictly between the "
                f"specified cutoff values. Given peak location: {self.peak}. "
                f"Given cutoffs: {cutoffs.to_tuple()}."
            )

    @property
    def margins(self) -> tuple[float, float]:
        """The left and right margin denoting the width of the triangle."""
        return self.peak - self.cutoffs.lower, self.cutoffs.upper - self.peak

    @classmethod
    def from_margins(
        cls, peak: float, margins: Sequence[float]
    ) -> TriangularTransformation:
        """Create a triangular transformation from a peak location and margins."""
        if len(margins) != 2:
            raise ValueError(
                "The margins must be provided as a sequence of two values."
            )
        return cls(peak=peak, cutoffs=Interval(peak - margins[0], peak + margins[1]))

    @classmethod
    def from_width(cls, peak: float, width: float) -> TriangularTransformation:
        """Create a triangular transformation from a peak location and width."""
        return cls.from_margins(peak, (width / 2, width / 2))

    @override
    def get_image(self, interval: Interval | None = None, /) -> Interval:
        return self._transformation.get_image(interval)

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return self._transformation(x)


@define(frozen=True)
class LogarithmicTransformation(MonotonicTransformation):
    """A logarithmic transformation."""

    @override
    def __call__(self, x: Tensor, /) -> Tensor:
        return x.log()


@define(frozen=True)
class ExponentialTransformation(MonotonicTransformation):
    """An exponential transformation."""

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


# Register (un-)structure hooks
converter.register_structure_hook(
    Transformation, get_base_structure_hook(Transformation)
)
converter.register_unstructure_hook(Transformation, unstructure_base)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
