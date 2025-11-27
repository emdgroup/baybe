"""Validation functionality for search spaces."""

import warnings
from collections.abc import Collection, Sequence
from typing import TypeVar

import pandas as pd

from baybe.exceptions import EmptySearchSpaceError, IncompatibilityError
from baybe.parameters import TaskParameter
from baybe.parameters.base import (
    Parameter,
    _DiscreteLabelLikeParameter,
)
from baybe.utils.dataframe import get_transform_objects

try:  # For python < 3.11, use the exceptiongroup backport
    ExceptionGroup
except NameError:
    from exceptiongroup import ExceptionGroup

_T = TypeVar("_T", bound=Parameter)


def validate_parameter_names(  # noqa: DOC101, DOC103
    parameters: Collection[Parameter],
) -> None:
    """Validate the parameter names.

    Raises:
        ValueError: If the given list contains parameters with the same name.
    """
    param_names = [p.name for p in parameters]
    if len(set(param_names)) != len(param_names):
        raise ValueError("All parameters must have unique names.")


def validate_parameters(parameters: Collection[Parameter]) -> None:  # noqa: DOC101, DOC103
    """Validate the parameters.

    Raises:
        EmptySearchSpaceError: If the parameter list is empty.
        NotImplementedError: If more than one
            :class:`baybe.parameters.categorical.TaskParameter` is requested.
    """
    if not parameters:
        raise EmptySearchSpaceError("At least one parameter must be provided.")

    # TODO [16932]: Remove once more task parameters are supported
    if len([p for p in parameters if isinstance(p, TaskParameter)]) > 1:
        raise NotImplementedError(
            "Currently, at most one task parameter can be considered."
        )

    # Assert: unique names
    validate_parameter_names(parameters)


def validate_dataframe_active_values(
    df: pd.DataFrame, parameters: Sequence[Parameter]
) -> None:
    """Validate that the dataframe is compatible with the active_values of parameters.

    Args:
        df: The dataframe to validate
        parameters: Sequence of parameters to check against

    Raises:
        IncompatibilityError: If dataframe contains values not in active_values
            of the parameters
        ExceptionGroup: If multiple parameters have validation errors
    """
    exceptions: list[Exception] = []

    for param in parameters:
        if isinstance(param, _DiscreteLabelLikeParameter):
            df_values = set(df[param.name].unique())
            active_values_set = set(param.active_values)
            if invalid_values := df_values - active_values_set:
                exceptions.append(
                    IncompatibilityError(
                        f"Dataframe column '{param.name}' contains invalid values "
                        f"{invalid_values}. Only active values {param.active_values} "
                        f"are allowed when using SearchSpace.from_dataframe."
                    )
                )
    if exceptions:
        if len(exceptions) == 1:
            raise exceptions[0]
        raise ExceptionGroup("Dataframe 'active_values' validation errors", exceptions)


def get_transform_parameters(
    parameters: Sequence[_T],
    df: pd.DataFrame,
    allow_missing: bool = False,
    allow_extra: bool = False,
) -> list[_T]:
    """Deprecated!"""  # noqa: D401
    warnings.warn(
        f"The function 'get_transform_parameters' has been deprecated and will be "
        f"removed in a future version. Use '{get_transform_objects.__name__}' instead.",
        DeprecationWarning,
    )
    return get_transform_objects(
        df, parameters, allow_missing=allow_missing, allow_extra=allow_extra
    )
