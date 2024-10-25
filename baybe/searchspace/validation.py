"""Validation functionality for search spaces."""

import warnings
from collections.abc import Collection, Sequence
from typing import TypeVar

import pandas as pd

from baybe.exceptions import EmptySearchSpaceError
from baybe.parameters import TaskParameter
from baybe.parameters.base import Parameter
from baybe.utils.dataframe import get_transform_objects

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
