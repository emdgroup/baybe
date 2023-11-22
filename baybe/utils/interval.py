"""Utilities for handling intervals."""

import sys
from collections.abc import Iterable
from functools import singledispatchmethod
from typing import Any, Union

import numpy as np
import torch
from attrs import define, field
from packaging import version

from baybe.utils.numeric import DTypeFloatNumpy, DTypeFloatTorch

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

    singledispatchmethod.register = _register


class InfiniteIntervalError(Exception):
    """An interval that should be finite is infinite."""


@define
class Interval:
    """Intervals on the real number line."""

    lower: float = field(converter=lambda x: float(x) if x is not None else -np.inf)
    """The lower end of the interval."""

    upper: float = field(converter=lambda x: float(x) if x is not None else np.inf)
    """The upper end of the interval."""

    @upper.validator
    def _validate_order(self, _: Any, value: float):  # noqa: DOC101, DOC103
        """Validate the order of the interval bounds.

        Raises:
            ValueError: If the upper end is not larger than the lower end.
        """
        if value <= self.lower:
            raise ValueError(
                f"The upper interval bound (provided value: {value}) must be larger "
                f"than the lower bound (provided value: {self.lower})."
            )

    @property
    def is_finite(self):
        """Check whether the interval is finite."""
        return np.isfinite(self.lower) and np.isfinite(self.upper)

    @property
    def is_bounded(self):
        """Check whether the interval is bounded."""
        return np.isfinite(self.lower) or np.isfinite(self.upper)

    @property
    def center(self):
        """The center of the interval. Only applicable for finite intervals."""
        if not self.is_finite:
            raise InfiniteIntervalError(
                f"The interval {self} is infinite and thus has no center."
            )
        return (self.lower + self.upper) / 2

    @singledispatchmethod
    @classmethod
    def create(cls, value):
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

    def to_tuple(self):
        """Transfor the interval to a tuple."""
        return self.lower, self.upper

    def to_ndarray(self):
        """Transform the interval to a ndarray."""
        return np.array([self.lower, self.upper], dtype=DTypeFloatNumpy)

    def to_tensor(self):
        """Transform the interval to a tensor."""
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
