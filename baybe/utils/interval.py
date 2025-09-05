"""Utilities for handling intervals."""

from __future__ import annotations

import builtins
import gc
from collections.abc import Iterable
from copy import deepcopy
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from attrs import define, field

from baybe.serialization import SerialMixin, converter
from baybe.utils.numerical import DTypeFloatNumpy
from baybe.utils.validation import non_nan_float

if TYPE_CHECKING:
    from torch import Tensor

# TODO[typing]: Add return type hints to classmethod constructors once ForwardRefs
#   are supported: https://bugs.python.org/issue41987


class InfiniteIntervalError(Exception):
    """An interval that should be finite is infinite."""


ConvertibleToInterval = Union["Interval", Iterable[float], None]
"""Types that can be converted to an :class:`Interval`."""


@define
class Interval(SerialMixin):
    """Intervals on the real number line."""

    lower: float = field(
        default=float("-inf"),
        converter=lambda x: float("-inf") if x is None else float(x),
        validator=non_nan_float,
    )
    """The lower end of the interval."""

    upper: float = field(
        default=float("inf"),
        converter=lambda x: float("inf") if x is None else float(x),
        validator=non_nan_float,
    )
    """The upper end of the interval."""

    @upper.validator
    def _validate_order(self, _: Any, upper: float) -> None:  # noqa: DOC101, DOC103
        """Validate the order of the interval bounds.

        Raises:
            ValueError: If the upper end is not larger than the lower end.
        """
        if upper < self.lower:
            raise ValueError(
                f"The upper interval bound (provided value: {upper}) cannot be smaller "
                f"than the lower bound (provided value: {self.lower})."
            )

    @property
    def is_degenerate(self) -> bool:
        """Check if the interval is degenerate (i.e., contains only a single number)."""
        return self.lower == self.upper

    @property
    def is_bounded(self) -> bool:
        """Check if the interval is bounded."""
        return self.is_left_bounded and self.is_right_bounded

    @property
    def is_left_bounded(self) -> bool:
        """Check if the interval is left-bounded."""
        return np.isfinite(self.lower)

    @property
    def is_right_bounded(self) -> bool:
        """Check if the interval is right-bounded."""
        return np.isfinite(self.upper)

    @property
    def is_half_bounded(self) -> bool:
        """Check if the interval is half-bounded."""
        return self.is_left_bounded ^ self.is_right_bounded

    @property
    def is_fully_unbounded(self) -> bool:
        """Check if the interval represents the entire real number line."""
        return not (self.is_left_bounded or self.is_right_bounded)

    @property
    def center(self) -> float:
        """The center of the interval, or ``nan`` if the interval is unbounded."""
        if not self.is_bounded:
            return float("nan")
        return (self.lower + self.upper) / 2

    @singledispatchmethod
    @classmethod
    def create(cls, value: ConvertibleToInterval) -> Interval:
        """Create an interval from various input types."""
        # Singledispatch does not play well with forward references, hence the
        # workaround via `isinstance` in the fallback method.
        # https://bugs.python.org/issue41987
        if isinstance(value, Interval):
            return deepcopy(value)

        raise NotImplementedError(f"Unsupported argument type: {type(value)}")

    @create.register
    @classmethod
    def _(cls, _: None):
        """Overloaded implementation for creating an unbounded interval."""
        return Interval()

    @create.register
    @classmethod
    def _(cls, bounds: Iterable):
        """Overloaded implementation for creating an interval from an iterable."""
        return Interval(*bounds)

    def clamp(self, min: float = float("-inf"), max: float = float("inf")) -> Interval:
        """Clamp the interval to a specified range."""
        return Interval(
            lower=builtins.max(self.lower, min),
            upper=builtins.min(self.upper, max),
        )

    def to_tuple(self) -> tuple[float, float]:
        """Transform the interval to a tuple."""
        return self.lower, self.upper

    def to_ndarray(self) -> np.ndarray:
        """Transform the interval to a :class:`numpy.ndarray`."""
        return np.array([self.lower, self.upper], dtype=DTypeFloatNumpy)

    def to_tensor(self) -> Tensor:
        """Transform the interval to a :class:`torch.Tensor`."""
        import torch

        from baybe.utils.torch import DTypeFloatTorch

        return torch.tensor([self.lower, self.upper], dtype=DTypeFloatTorch)

    def contains(self, number: float) -> bool:
        """Check whether the interval contains a given number.

        Args:
            number: The number that should be checked.

        Returns:
            Whether the interval contains the number.
        """
        return (
            bool(np.isclose(number, self.lower))
            or bool(np.isclose(number, self.upper))
            or (self.lower < number < self.upper)
        )


def use_fallback_constructor_hook(value: Any, cls: type[Interval]) -> Interval:
    """Use the singledispatch mechanism as fallback to parse arbitrary input."""
    if isinstance(value, dict):
        return converter.structure_attrs_fromdict(value, cls)
    return Interval.create(value)


# Register structure hooks
converter.register_structure_hook(Interval, use_fallback_constructor_hook)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
