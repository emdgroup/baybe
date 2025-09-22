"""Functionality for constraint conditions."""

from __future__ import annotations

import gc
import operator as ops
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from attrs import define, field
from attrs.converters import optional as optional_c
from attrs.validators import ge, in_, min_len
from attrs.validators import optional as optional_v
from numpy.typing import ArrayLike
from typing_extensions import override

from baybe.parameters.validation import validate_unique_values
from baybe.serialization import (
    SerialMixin,
)
from baybe.utils.basic import to_tuple
from baybe.utils.numerical import DTypeFloatNumpy
from baybe.utils.validation import finite_float

if TYPE_CHECKING:
    import polars as pl


def _is_not_close(x: ArrayLike, y: ArrayLike, rtol: float, atol: float) -> np.ndarray:
    """Return a Boolean array indicating where ``x`` and ``y`` are not close.

    The counterpart to ``numpy.isclose``.

    Args:
        x: First input array to compare.
        y: Second input array to compare.
        rtol: The relative tolerance parameter.
        atol: The absolute tolerance parameter.

    Returns:
        A Boolean array of where ``x`` and ``y`` are not equal within the
        given tolerances.

    """
    return np.logical_not(_is_close(x, y, rtol=rtol, atol=atol))


def _is_close(x: ArrayLike, y: ArrayLike, rtol: float, atol: float) -> np.ndarray:
    """Return a Boolean array indicating where ``x`` and ``y`` are close.

    The equivalent to :func:``numpy.isclose``.
    Using ``numpy.isclose`` with Polars dataframes results in this error:
    ``TypeError: ufunc 'isfinite' not supported for the input types``.

    Args:
        x: First input array to compare.
        y: Second input array to compare.
        rtol: The relative tolerance parameter.
        atol: The absolute tolerance parameter.

    Returns:
        A Boolean array of where ``x`` and ``y`` are equal within the
        given tolerances.

    """
    return np.abs(np.subtract(x, y)) <= atol + rtol * np.abs(y)


# provide threshold operators
_threshold_operators: dict[str, Callable] = {
    "<": ops.lt,
    "<=": ops.le,
    "=": partial(_is_close, rtol=0.0),
    "==": partial(_is_close, rtol=0.0),
    "!=": partial(_is_not_close, rtol=0.0),
    ">": ops.gt,
    ">=": ops.ge,
}

# define operators that are eligible for tolerance
_valid_tolerance_operators = ["=", "==", "!="]

_valid_logic_combiners = {
    "AND": ops.and_,
    "OR": ops.or_,
    "XOR": ops.xor,
}


class Condition(ABC, SerialMixin):
    """Abstract base class for all conditions.

    Conditions always evaluate an expression regarding a single parameter.
    Conditions are part of constraints, a constraint can have multiple conditions.
    """

    @abstractmethod
    def evaluate(self, data: pd.Series) -> pd.Series:
        """Evaluate the condition on a given data series.

        Args:
            data: A series containing parameter values.

        Returns:
            A Boolean series indicating which elements satisfy the condition.
        """

    @abstractmethod
    def to_polars(self, expr: pl.Expr, /) -> pl.Expr:
        """Apply the condition to a Polars expression.

        Args:
            expr: Input expression, for instance column selection etc.

        Returns:
            An expression that can be used for filtering.
        """


@define
class ThresholdCondition(Condition):
    """Class for modelling threshold-based conditions."""

    # object variables
    threshold: float = field(converter=float, validator=finite_float)
    """The threshold value used in the condition."""

    operator: str = field(validator=[in_(_threshold_operators)])
    """The operator used in the condition."""

    tolerance: float | None = field(
        converter=optional_c(float), validator=optional_v([finite_float, ge(0)])
    )
    """A numerical tolerance. Set to a reasonable default tolerance."""

    @tolerance.default
    def _tolerance_default(self) -> float | None:
        """Create the default value for the tolerance."""
        # Default value for the tolerance.
        return 1e-8 if self.operator in _valid_tolerance_operators else None

    @tolerance.validator
    def _validate_tolerance(self, _: Any, value: float) -> None:  # noqa: DOC101, DOC103
        """Validate the threshold condition tolerance.

        Raises:
            ValueError: If the operator does not allow for setting a tolerance.
            ValueError: If the operator allows for setting a tolerance, but the provided
                tolerance is either less than 0 or ``None``.
        """
        if (self.operator not in _valid_tolerance_operators) and (value is not None):
            raise ValueError(
                f"Setting the tolerance for a threshold condition is only valid "
                f"with the following operators: {_valid_tolerance_operators}."
            )
        if self.operator in _valid_tolerance_operators:
            if (value is None) or (value <= 0.0):
                raise ValueError(
                    f"When using a tolerance-enabled operator"
                    f" ({_valid_tolerance_operators}) the tolerance cannot be None "
                    f"or <= 0.0, but was {value}."
                )

    def _make_operator_function(self):
        """Generate a function using operators to filter out undesired rows."""

        def evaluate(x: ArrayLike, /, **kwargs) -> Callable:
            """Evaluate the condition on a given input."""
            return _threshold_operators[self.operator](x, self.threshold, **kwargs)

        if self.operator in _valid_tolerance_operators:
            return partial(evaluate, atol=self.tolerance)

        return evaluate

    @override
    def evaluate(self, data: pd.Series) -> pd.Series:
        if data.dtype.kind not in "iufb":
            raise ValueError(
                "You tried to apply a threshold condition to non-numeric data. "
                "This operation is error-prone and not supported. Only use threshold "
                "conditions with numerical parameters."
            )
        func = self._make_operator_function()
        return data.apply(func)

    @override
    def to_polars(self, expr: pl.Expr, /) -> pl.Expr:
        op = self._make_operator_function()
        return op(expr)


@define
class SubSelectionCondition(Condition):
    """Class for defining valid parameter entries."""

    # object variables
    _selection: tuple = field(
        converter=to_tuple,
        # FIXME[typing]: https://github.com/python-attrs/attrs/issues/1197
        validator=[
            min_len(1),
            validate_unique_values,  # type: ignore
        ],
    )
    """The internal list of items which are considered valid."""

    @property
    def selection(self) -> tuple:
        """The list of items which are considered valid."""
        return tuple(
            DTypeFloatNumpy(itm) if isinstance(itm, (float, int, bool)) else itm
            for itm in self._selection
        )

    @override
    def evaluate(self, data: pd.Series) -> pd.Series:
        return data.isin(self.selection)

    @override
    def to_polars(self, expr: pl.Expr, /) -> pl.Expr:
        return expr.is_in(self.selection)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
