"""Functionality for constraint conditions."""

import operator as ops
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from attr import define, field
from attr.validators import in_
from funcy import rpartial
from numpy.typing import ArrayLike

from baybe.utils import SerialMixin


def _is_not_close(x: ArrayLike, y: ArrayLike, rtol: float, atol: float) -> np.ndarray:
    """Return a boolean array indicating where ``x`` and ``y`` are not close.

    The counterpart to ``numpy.isclose``.

    Args:
        x: First input array to compare.
        y: Second input array to compare.
        rtol: The relative tolerance parameter.
        atol: The absolute tolerance parameter.

    Returns:
        Returns a boolean array of where ``x`` and ``y`` are not equal within the
        given tolerances.

    """
    return np.logical_not(np.isclose(x, y, rtol=rtol, atol=atol))


# provide threshold operators
_threshold_operators = {
    "<": ops.lt,
    "<=": ops.le,
    "=": rpartial(np.isclose, rtol=0.0),
    "==": rpartial(np.isclose, rtol=0.0),
    "!=": rpartial(_is_not_close, rtol=0.0),
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
            A boolean series indicating which elements satisfy the condition.
        """


@define
class ThresholdCondition(Condition):
    """Class for modelling threshold-based conditions."""

    # object variables
    threshold: float = field()
    """The threshold value used in the condition."""

    operator: str = field(validator=[in_(_threshold_operators)])
    """The operator used in the condition."""

    tolerance: Optional[float] = field()
    """A numerical tolerance. Set to a reasonable default tolerance."""

    @tolerance.default
    def _tolerance_default(self) -> Union[float, None]:
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

    def evaluate(self, data: pd.Series) -> pd.Series:  # noqa: D102
        # See base class.
        if data.dtype.kind not in "iufb":
            raise ValueError(
                "You tried to apply a threshold condition to non-numeric data. "
                "This operation is error-prone and not supported. Only use threshold "
                "conditions with numerical parameters."
            )
        func = rpartial(_threshold_operators[self.operator], self.threshold)
        if self.operator in _valid_tolerance_operators:
            func = rpartial(func, atol=self.tolerance)

        return data.apply(func)


@define
class SubSelectionCondition(Condition):
    """Class for defining valid parameter entries."""

    # object variables
    selection: List[Any] = field()
    """The list of items which are considered valid."""

    def evaluate(self, data: pd.Series) -> pd.Series:  # noqa: D102
        # See base class.
        return data.isin(self.selection)
