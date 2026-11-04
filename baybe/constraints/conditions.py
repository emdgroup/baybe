"""Functionality for constraint conditions."""

from __future__ import annotations

import gc
import numbers
import operator as ops
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import narwhals.stable.v2 as nw
from attrs import define, field
from attrs.converters import optional as optional_c
from attrs.validators import ge, in_, min_len
from attrs.validators import optional as optional_v
from typing_extensions import override

from baybe.serialization import (
    SerialMixin,
)
from baybe.utils.basic import to_tuple
from baybe.utils.validation import finite_float, validate_unique_values

if TYPE_CHECKING:
    from narwhals.stable.v2 import Expr, Series
    from narwhals.stable.v2.typing import IntoSeries, IntoSeriesT


ThresholdOperator: TypeAlias = Literal["<", "<=", "=", "==", "!=", ">", ">="]
"""An operator for comparing numerical values against a threshold."""

_THRESHOLD_OPERATORS: dict[ThresholdOperator, Callable] = {
    "<": ops.lt,
    "<=": ops.le,
    "=": ops.eq,
    "==": ops.eq,
    "!=": ops.ne,
    ">": ops.gt,
    ">=": ops.ge,
}
_TOLERANCE_OPERATORS = ["=", "==", "!="]
_LOGIC_COMBINERS = {
    "AND": ops.and_,
    "OR": ops.or_,
    "XOR": ops.ne,
}


def _selection_dtypes(value: Any) -> tuple[type, ...]:
    """Return the Narwhals dtypes that a sub-selection entry can be matched against."""
    if isinstance(value, bool):
        return (nw.dtypes.Boolean,)
    if isinstance(value, numbers.Real):
        return (nw.dtypes.NumericType,)
    if isinstance(value, str):
        return nw.dtypes.String, nw.dtypes.Categorical, nw.dtypes.Enum
    raise ValueError(
        f"Sub-selection entries must be strings, Booleans, or numbers, but got "
        f"{value!r}."
    )


class Condition(ABC, SerialMixin):
    """Abstract base class for all conditions.

    Conditions always evaluate an expression regarding a single parameter.
    Conditions are part of :class:`~baybe.constraints.base.Constraint` and
    :class:`~baybe.symmetries.base.Symmetry` objects.
    """

    def __call__(self, series: IntoSeriesT, /) -> IntoSeriesT:
        """Apply the condition and return the values that satisfy it.

        Args:
            series: A native series containing the values to be filtered.

        Returns:
            A native series containing only the values that satisfy the condition.

        Raises:
            ValueError: If the input series is incompatible with the condition.
        """
        nw_series = nw.from_native(series, series_only=True)
        mask = nw.from_native(self.evaluate(series), series_only=True)
        return nw_series.filter(mask).to_native()

    def evaluate(self, series: IntoSeries, /) -> IntoSeries:
        """Evaluate the condition and return the corresponding Boolean series.

        Args:
            series: A native series containing the values to be evaluated.

        Returns:
            A native Boolean series indicating which values satisfy the condition.

        Raises:
            ValueError: If the input series is incompatible with the condition.
        """
        nw_series = nw.from_native(series, series_only=True)
        self._validate_series(nw_series)
        nw_frame = nw_series.to_frame()
        column = nw_frame.columns[0]
        result = nw_frame.select(self._to_expr(nw.col(column)))
        return result.get_column(result.columns[0]).alias(nw_series.name).to_native()

    @abstractmethod
    def _to_expr(self, expr: Expr, /) -> Expr:
        """Return the condition as a Narwhals expression.

        Args:
            expr: Input expression, for instance column selection etc.

        Returns:
            A backend-agnostic Boolean expression.
        """

    def _validate_series(self, series: Series, /) -> None:
        """Validate the input series before applying the condition."""


@define
class ThresholdCondition(Condition):
    """Class for modelling threshold-based conditions."""

    # object variables
    threshold: float = field(converter=float, validator=finite_float)
    """The threshold value used in the condition."""

    operator: ThresholdOperator = field(validator=[in_(_THRESHOLD_OPERATORS)])
    """The operator used in the condition."""

    tolerance: float | None = field(
        converter=optional_c(float), validator=optional_v([finite_float, ge(0)])
    )
    """A numerical tolerance. Set to a reasonable default tolerance."""

    @tolerance.default
    def _tolerance_default(self) -> float | None:
        """Create the default value for the tolerance."""
        return 1e-8 if self.operator in _TOLERANCE_OPERATORS else None

    @tolerance.validator
    def _validate_tolerance(self, _: Any, value: float) -> None:  # noqa: DOC101, DOC103
        """Validate the threshold condition tolerance.

        Raises:
            ValueError: If the operator does not allow for setting a tolerance.
            ValueError: If the operator allows for setting a tolerance, but the provided
                tolerance is either less than 0 or ``None``.
        """
        if (self.operator not in _TOLERANCE_OPERATORS) and (value is not None):
            raise ValueError(
                f"Setting the tolerance for a threshold condition is only valid "
                f"with the following operators: {_TOLERANCE_OPERATORS}."
            )
        if self.operator in _TOLERANCE_OPERATORS:
            if (value is None) or (value <= 0.0):
                raise ValueError(
                    f"When using a tolerance-enabled operator"
                    f" ({_TOLERANCE_OPERATORS}) the tolerance cannot be None "
                    f"or <= 0.0, but was {value}."
                )

    @override
    def _validate_series(self, series: Series, /) -> None:
        if not isinstance(series.dtype, (nw.dtypes.NumericType, nw.dtypes.Boolean)):
            raise ValueError(
                "You tried to apply a threshold condition to non-numeric data. "
                "This operation is error-prone and not supported. Only use threshold "
                "conditions with numerical parameters."
            )

    @override
    def _to_expr(self, expr: Expr, /) -> Expr:
        if self.operator in _TOLERANCE_OPERATORS:
            assert self.tolerance is not None
            close = (expr >= self.threshold - self.tolerance) & (
                expr <= self.threshold + self.tolerance
            )
            matching = ~close if self.operator == "!=" else close
        else:
            matching = _THRESHOLD_OPERATORS[self.operator](expr, self.threshold)

        return (matching & ~expr.is_null()).fill_null(False)


@define
class SubSelectionCondition(Condition):
    """Class for defining valid parameter entries."""

    # object variables
    selection: tuple = field(
        converter=to_tuple,
        # FIXME[typing]: https://github.com/python-attrs/attrs/issues/1197
        validator=[
            min_len(1),
            validate_unique_values,
        ],
    )
    """The items that are considered valid."""

    @selection.validator
    def _validate_selection(self, _: Any, selection: tuple) -> None:
        dtypes = _selection_dtypes(selection[0])
        if any(_selection_dtypes(item) != dtypes for item in selection[1:]):
            raise ValueError(
                f"All entries of a '{type(self).__name__}' must have a common type, "
                f"but got {selection!r}."
            )

    @property
    def _dtypes(self) -> tuple[type, ...]:
        """The Narwhals dtypes that the selection entries can be matched against."""
        return _selection_dtypes(self.selection[0])

    @override
    def _validate_series(self, series: Series, /) -> None:
        # All-null series are typeless, and the backends report different dtypes for
        # them, so no compatibility check is possible.
        if isinstance(series.dtype, self._dtypes) or series.is_null().all():
            return

        raise ValueError(
            f"A '{type(self).__name__}' with entries {self.selection!r} cannot "
            f"be applied to data of type '{series.dtype}'."
        )

    @override
    def _to_expr(self, expr: Expr, /) -> Expr:
        return expr.is_in(self.selection).fill_null(False)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
