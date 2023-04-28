# pylint: disable=missing-function-docstring, missing-module-docstring

import sys
from functools import singledispatchmethod

import numpy as np
import torch
from attrs import define, field
from packaging import version

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
    upper: float = field(converter=lambda x: float(x) if x is not None else np.inf)

    @upper.validator
    def validate_order(self, _, value):
        if value < self.lower:
            raise ValueError(
                f"The upper interval bound (provided value: {value}) must be larger "
                f"than the lower bound (provided value: {self.lower})."
            )

    @property
    def is_finite(self):
        return np.isfinite(self.lower) and np.isfinite(self.upper)

    @property
    def is_bounded(self):
        return np.isfinite(self.lower) or np.isfinite(self.upper)

    @property
    def center(self):
        if not self.is_finite:
            raise InfiniteIntervalError(
                f"The interval {self} is infinite and thus has no center."
            )
        return (self.lower + self.upper) / 2

    @singledispatchmethod
    @classmethod
    def create(cls, value):
        raise NotImplementedError(f"Unsupported argument type: {type(value)}")

    @create.register
    @classmethod
    def _(cls, none: None):  # pylint: disable=unused-argument
        return Interval(-np.inf, np.inf)

    @create.register
    @classmethod
    def _(cls, bounds: tuple):
        return Interval(*bounds)

    def to_tuple(self):
        return self.lower, self.upper

    def to_ndarray(self):
        return np.array([self.lower, self.upper])

    def to_tensor(self):
        return torch.tensor([self.lower, self.upper])
