"""Utilities for handling intervals."""

import sys
import warnings
from collections.abc import Iterable
from functools import singledispatchmethod
from typing import Any, Optional, Tuple, Type, Union

import numpy as np
import torch
from attrs import define, field
from packaging import version

from baybe.serialization import SerialMixin, converter
from baybe.utils.numerical import DTypeFloatNumpy, DTypeFloatTorch

# TODO[typing]: Add return type hints to classmethod constructors once ForwardRefs
#   are supported: https://bugs.python.org/issue41987

# TODO: Remove when upgrading python version
if version.parse(sys.version.split()[0]) < version.parse("3.9.8"):
    # Monkeypatching necessary due to functools bug fixed in 3.9.8
    #   https://stackoverflow.com/questions/62696796/singledispatchmethod-and-
    #       class-method-decorators-in-python-3-8
    #   https://bugs.python.org/issue39679
    def _register(self, cls, method=None):
        if hasattr(cls, "__func__"):
            setattr(cls, "__annotations__", cls.__func__.__annotations__)
        return self.dispatcher.register(cls, func=method)

    singledispatchmethod.register = _register  # type: ignore[method-assign]


class InfiniteIntervalError(Exception):
    """An interval that should be finite is infinite."""


@define
class Interval(SerialMixin):
    """Intervals on the real number line."""

    lower: float = field(converter=lambda x: float(x) if x is not None else -np.inf)
    """The lower end of the interval."""

    upper: float = field(converter=lambda x: float(x) if x is not None else np.inf)
    """The upper end of the interval."""

    @upper.validator
    def _validate_order(self, _: Any, upper: float) -> None:  # noqa: DOC101, DOC103
        """Validate the order of the interval bounds.

        Raises:
            ValueError: If the upper end is not larger than the lower end.
        """
        if upper <= self.lower:
            raise ValueError(
                f"The upper interval bound (provided value: {upper}) must be larger "
                f"than the lower bound (provided value: {self.lower})."
            )

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
    def is_finite(self) -> bool:
        """Check whether the interval is finite."""
        warnings.warn(
            "The use of 'Interval.is_finite' is deprecated and will be disabled in "
            "a future version. Use 'Interval.is_bounded' instead.",
            DeprecationWarning,
        )
        return np.isfinite(self.lower) and np.isfinite(self.upper)

    @property
    def center(self) -> Optional[float]:
        """The center of the interval, or ``None`` if the interval is unbounded."""
        if not self.is_bounded:
            return None
        return (self.lower + self.upper) / 2

    @singledispatchmethod
    @classmethod
    def create(cls, value: Any):
        """Create an interval from various input types."""
        raise NotImplementedError(f"Unsupported argument type: {type(value)}")

    @create.register
    @classmethod
    def _(cls, _: None):
        """Overloaded implementation for creating an empty interval."""
        return Interval(-np.inf, np.inf)

    @create.register
    @classmethod
    def _(cls, bounds: Iterable):
        """Overloaded implementation for creating an interval of an iterable."""
        return Interval(*bounds)

    def to_tuple(self) -> Tuple[float, float]:
        """Transform the interval to a tuple."""
        return self.lower, self.upper

    def to_ndarray(self) -> np.ndarray:
        """Transform the interval to a :class:`numpy.ndarray`."""
        return np.array([self.lower, self.upper], dtype=DTypeFloatNumpy)

    def to_tensor(self) -> torch.Tensor:
        """Transform the interval to a :class:`torch.Tensor`."""
        return torch.tensor([self.lower, self.upper], dtype=DTypeFloatTorch)

    def contains(self, number: float) -> bool:
        """Check whether the interval contains a given number.

        Args:
            number: The number that should be checked.

        Returns:
            Whether or not the interval contains the number.
        """
        return self.lower <= number <= self.upper


def convert_bounds(bounds: Union[None, tuple, Interval]) -> Interval:
    """Convert bounds given in another format to an interval.

    Args:
        bounds: The bounds that should be transformed to an interval.

    Returns:
        The interval.
    """
    if isinstance(bounds, Interval):
        return bounds
    return Interval.create(bounds)


def use_fallback_constructor_hook(value: Any, cls: Type[Interval]) -> Interval:
    """Use the singledispatch mechanism as fallback to parse arbitrary input."""
    if isinstance(value, dict):
        return converter.structure_attrs_fromdict(value, cls)
    return Interval.create(value)


# Register structure hooks
converter.register_structure_hook(Interval, use_fallback_constructor_hook)
