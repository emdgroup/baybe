"""Validation functionality for search spaces."""

from typing import List

from baybe.exceptions import EmptySearchSpaceError
from baybe.parameters import TaskParameter
from baybe.parameters.base import Parameter


def validate_parameter_names(  # noqa: DOC101, DOC103
    parameters: List[Parameter],
) -> None:
    """Validate the parameter names.

    Raises:
        ValueError: If the given list contains parameters with the same name.
    """
    param_names = [p.name for p in parameters]
    if len(set(param_names)) != len(param_names):
        raise ValueError("All parameters must have unique names.")


def validate_parameters(parameters: List[Parameter]) -> None:  # noqa: DOC101, DOC103
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
