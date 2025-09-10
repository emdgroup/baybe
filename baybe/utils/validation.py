"""Validation utilities."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from attrs import Attribute

from baybe.exceptions import IncompleteMeasurementsError

if TYPE_CHECKING:
    from baybe.objectives.base import Objective
    from baybe.parameters.base import Parameter
    from baybe.targets.base import Target


def validate_not_nan(self: Any, attribute: Attribute, value: Any) -> None:
    """Attrs-compatible validator to forbid 'nan' values."""
    if isinstance(value, float) and math.isnan(value):
        raise ValueError(
            f"The value passed to attribute '{attribute.name}' of class "
            f"'{self.__class__.__name__}' cannot be 'nan'."
        )


def _make_restricted_float_validator(
    allow_nan: bool, allow_inf: bool
) -> Callable[[Any, Attribute, Any], None]:
    """Make an attrs-compatible validator for restricted floats.

    Args:
        allow_nan: If False, validated values cannot be 'nan'.
        allow_inf: If False, validated values cannot be 'inf' or '-inf'.

    Raises:
        ValueError: If no float range restriction is in place.

    Returns:
        The validator.
    """
    if allow_nan and allow_inf:
        raise ValueError(
            "The requested validator would not restrict the float range. "
            "Hence, you can use `attrs.validators.instance_of(float)` instead."
        )

    def validator(self: Any, attribute: Attribute, value: Any) -> None:
        if not isinstance(value, float):
            raise ValueError(
                f"Values assigned to attribute '{attribute.name}' of class "
                f"'{self.__class__.__name__}' must be of type 'float'. "
                f"Given: {value} (type: {type(value)})"
            )
        if not allow_inf and math.isinf(value):
            raise ValueError(
                f"Values assigned to attribute '{attribute.name}' of class "
                f"'{self.__class__.__name__}' cannot be 'inf' or '-inf'."
            )
        if not allow_nan and math.isnan(value):
            raise ValueError(
                f"Values assigned to attribute '{attribute.name}' of class "
                f"'{self.__class__.__name__}' cannot be 'nan'."
            )

    return validator


finite_float = _make_restricted_float_validator(allow_nan=False, allow_inf=False)
"""Validator for finite (i.e., non-nan and non-infinite) floats."""

non_nan_float = _make_restricted_float_validator(allow_nan=False, allow_inf=True)
"""Validator for non-nan floats."""

non_inf_float = _make_restricted_float_validator(allow_nan=True, allow_inf=False)
"""Validator for non-infinite floats."""


def validate_target_input(data: pd.DataFrame, targets: Iterable[Target]) -> None:
    """Validate input dataframe columns corresponding to targets.

    Args:
        data: The input dataframe to be validated.
        targets: The allowed targets.

    Raises:
        ValueError: If the data is empty.
        ValueError: If the data misses columns for a target.
        TypeError: If any numerical target data contain non-numeric values.
        ValueError: If any binary target data contain values not part of the targets'
            allowed values or NaN.
    """
    from baybe.targets import BinaryTarget, NumericalTarget

    if data.empty:
        raise ValueError("The provided input dataframe cannot be empty.")

    if missing := {t.name for t in targets}.difference(data.columns):
        raise ValueError(
            f"The input dataframe is missing columns for the following targets: "
            f"{missing}"
        )

    for t in targets:
        if isinstance(t, NumericalTarget):
            if data[t.name].dtype.kind not in "iufb":
                raise TypeError(
                    f"The numerical target '{t.name}' has non-numeric entries in the "
                    f"provided dataframe."
                )
        elif isinstance(t, BinaryTarget):
            allowed = {t.failure_value, t.success_value, np.nan}
            if invalid := set(data[t.name].unique()) - allowed:
                raise ValueError(
                    f"The binary target '{t.name}' has invalid entries {invalid} "
                    f"in the provided dataframe. Allowed values are: {allowed}."
                )


def validate_objective_input(data: pd.DataFrame, objective: Objective) -> None:
    """Validate the input dataframe for the given objective.

    Args:
        data: The input dataframe to be validated.
        objective: The objective to validate against.

    Raises:
        IncompleteMeasurementsError: If the objective requires complete measurements
            but the input dataframe contains missing target values.
    """
    data = data[[t.name for t in objective.targets]]
    if not objective.supports_partial_measurements and (
        cols := data.columns[data.isna().any()].tolist()
    ):
        raise IncompleteMeasurementsError(
            f"The used objective requires complete measurements of all "
            f"involved targets but the provided dataframe contains missing "
            f"values for the following targets: {cols}"
        )


def validate_parameter_input(
    data: pd.DataFrame,
    parameters: Iterable[Parameter],
    numerical_measurements_must_be_within_tolerance: bool = False,
) -> None:
    """Validate input dataframe columns corresponding to parameters.

    Args:
        data: The input dataframe to be validated.
        parameters: The allowed parameters.
        numerical_measurements_must_be_within_tolerance: If ``True``, numerical
            parameter values must match to parameter values within the
            parameter-specific tolerance.

    Raises:
        ValueError: If the data is empty.
        ValueError: If the data misses columns for a parameter.
        ValueError: If a parameter contains NaN.
        TypeError: If a parameter contains non-numeric values.
    """
    if data.empty:
        raise ValueError("The provided input dataframe cannot be empty.")

    if missing := {p.name for p in parameters}.difference(data.columns):
        raise ValueError(
            f"The input dataframe is missing columns for the following parameters: "
            f"{missing}"
        )

    for p in parameters:
        if data[p.name].isna().any():
            raise ValueError(
                f"The parameter '{p.name}' has missing values in the provided "
                f"dataframe."
            )
        if p.is_numerical and (data[p.name].dtype.kind not in "iufb"):
            raise TypeError(
                f"The numerical parameter '{p.name}' has non-numeric entries in the "
                f"provided dataframe."
            )

        # Check if all rows have valid inputs matching allowed parameter values
        for ind, row in data.iterrows():
            valid = True
            if p.is_numerical:
                if numerical_measurements_must_be_within_tolerance:
                    valid &= p.is_in_range(row[p.name])
            else:
                valid &= p.is_in_range(row[p.name])
            if not valid:
                raise ValueError(
                    f"Input data on row with the index {row.name} has invalid "
                    f"values in parameter '{p.name}'. "
                    f"For categorical parameters, values need to exactly match a "
                    f"valid choice defined in your config. "
                    f"For numerical parameters, a match is accepted only if "
                    f"the input value is within the specified tolerance/range. Set "
                    f"the flag 'numerical_measurements_must_be_within_tolerance' "
                    f"to 'False' to disable this behavior."
                )


def validate_object_names(objects: Iterable[Parameter | Target], /) -> None:
    """Validate that the provided objects have unique names.

    Args:
        objects: An iterable containing a combination of parameters and targets.

    Raises:
        ValueError: If two or more objects have the same name.
    """
    names = [obj.name for obj in objects]
    if len(names) != len(set(names)):
        duplicates = {name for name in names if names.count(name) > 1}
        raise ValueError(
            f"All parameters and targets must have unique names. The following names "
            f"appear multiple times: {duplicates}."
        )
