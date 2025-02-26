"""Parameter utilities."""

from collections.abc import Callable, Collection
from typing import Any, TypeVar

import pandas as pd
from attrs import evolve

from baybe.parameters.base import Parameter
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.utils.interval import Interval

_TParameter = TypeVar("_TParameter", bound=Parameter)


def get_parameters_from_dataframe(
    df: pd.DataFrame,
    factory: Callable[[str, Collection[Any]], _TParameter],
    parameters: Collection[_TParameter] | None = None,
) -> list[_TParameter]:
    """Create a list of parameters from a dataframe.

    Returns one parameter for each column of the given dataframe. By default,
    the parameters are created using the provided factory, which takes the name
    of the column and its unique values as arguments. However, there is also
    the possibility to provide explicit parameter objects with names matching specific
    columns of the dataframe, to bypass the parameter factory creation for those
    columns. This allows finer control, for example, to specify custom parameter
    attributes (e.g. specific optional arguments) compared to what would be provided
    by the factory. Still, the pre-specified parameters are validated to ensure that
    they are compatible with the contents of the dataframe.

    Args:
        df: The dataframe from which to create the parameters.
        factory: A parameter factor, creating parameter objects for the columns
            from the column name and the unique column values.
        parameters: An optional list of parameter objects to bypass the factory
            creation for columns whose names match with the parameter names.

    Returns:
        The combined parameter list, containing both the (validated) pre-specified
        parameters and the parameters inferred from the dataframe.

    Raises:
        ValueError: If several parameters with identical names are provided.
        ValueError: If a parameter was specified for which no match was found.
    """
    # Turn the pre-specified parameters into a dict and check for duplicate names
    specified_params: dict[str, _TParameter] = {}
    if parameters is not None:
        for param in parameters:
            if param.name in specified_params:
                raise ValueError(
                    f"You provided several parameters with the name '{param.name}'."
                )
            specified_params[param.name] = param

    # Try to find a parameter match for each dataframe column
    parameters = []
    for name, series in df.items():
        assert isinstance(
            name, str
        ), "The given dataframe must only contain string-valued column names."
        unique_values = series.unique()

        # If a match is found, assert that the values are in range
        if match := specified_params.pop(name, None):
            if not all(match.is_in_range(x) for x in unique_values):
                raise ValueError(
                    f"The dataframe column '{name}' contains the values "
                    f"{unique_values}, which are outside the range of {match}."
                )
            parameters.append(match)

        # Otherwise, create a new parameter using the factory
        else:
            param = factory(name, unique_values)
            parameters.append(param)

    # By now, all pre-specified parameters must have been used
    if specified_params:
        raise ValueError(
            f"For the parameter(s) {list(specified_params.keys())}, "
            f"no match could be found in the given dataframe."
        )

    return parameters


def sort_parameters(parameters: Collection[Parameter]) -> tuple[Parameter, ...]:
    """Sort parameters alphabetically by their names."""
    return tuple(sorted(parameters, key=lambda p: p.name))


def activate_parameter(
    parameter: NumericalContinuousParameter, thresholds: Interval
) -> NumericalContinuousParameter:
    """Force-activates a given parameter by moving its bounds away from zero.

    A parameter that is trivially active because its value range does not overlap
    with the considered inactivity interval is unaffected.

    Important:
        A parameter whose range includes zero but extends beyond the threshold interval
        on both sides remains unchanged, because the corresponding activated parameter
        would no longer have a continuous value range.

    Args:
        parameter: The parameter to be activated.
        thresholds: The considered parameter (in)activity thresholds.

    Returns:
        A copy of the parameter with adjusted bounds.

    Raises:
        ValueError: If the threshold interval does not contain zero.
        ValueError: If the parameter cannot be activated since both its bounds are
            in the inactive range.
    """
    lower_bound = parameter.bounds.lower
    upper_bound = parameter.bounds.upper

    if not thresholds.contains(0.0):
        raise ValueError(
            f"The thresholds must cover zero but ({thresholds.lower}, "
            f"{thresholds.upper}) was given."
        )

    def in_inactive_range(x: float) -> bool:
        """Return true when x is within the inactive range."""
        return thresholds.lower <= x <= thresholds.upper

    # When both bounds are in the in inactive range
    if in_inactive_range(lower_bound) and in_inactive_range(upper_bound):
        raise ValueError(
            f"Parameter '{parameter.name}' cannot be set active since its "
            f"bounds {parameter.bounds.to_tuple()} are entirely contained in the "
            f"inactive range ({thresholds.lower}, {thresholds.upper})."
        )

    # When the upper bound is in inactive range, move it to the lower threshold of the
    # inactive region
    if lower_bound < thresholds.lower and in_inactive_range(upper_bound):
        return evolve(parameter, bounds=(lower_bound, thresholds.lower))

    # When the lower bound is in inactive range, move it to the upper threshold of
    # the inactive region
    if upper_bound > thresholds.upper and in_inactive_range(lower_bound):
        return evolve(parameter, bounds=(thresholds.upper, upper_bound))

    # When the parameter is already trivially active (or activating it would tear
    # its value range apart)
    return parameter
