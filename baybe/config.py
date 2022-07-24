# pylint: disable = no-self-use, no-self-argument
"""
Config functionality
"""

import logging
from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, validator

from baybe import parameters, targets
from baybe.parameters import GenericParameter

log = logging.getLogger(__name__)


def check_if_in(element: Any, allowed: list):
    """
    Checks if an element is in a given list of elements and raises a
    context-specific exception if it is not.
    """
    if element not in allowed:
        raise ValueError(
            f"The value '{element}' is not allowed. Must be one of {allowed}."
        )


class ParameterConfig(BaseModel):
    """Configuration class for creating parameter objects."""

    name: str
    type: str
    values: list
    tolerance: Optional[float]
    encoding: Optional[str]

    class Config:
        """Pydantic configuration."""

        extra = "forbid"

    @validator("type")
    def validate_type(cls, val):
        """Validates if the given parameter type exists."""
        check_if_in(val, GenericParameter.SUBCLASSES)
        return val

    @validator("encoding")
    def validate_encoding(cls, val, values):
        """Validates if the given encoding exists for the selected parameter type."""
        check_if_in(val, GenericParameter.ENCODINGS[values["type"]])
        return val


class TargetConfig(BaseModel):
    """Configuration class for creating target objects."""

    name: str
    type: str
    mode: str
    bounds: Optional[str]

    class Config:
        """Pydantic configuration."""

        extra = "forbid"


class ObjectiveConfig(BaseModel):
    """Configuration class for creating objective objects."""

    mode: str
    targets: List[dict]

    class Config:
        """Pydantic configuration."""

        extra = "forbid"

    @validator("targets")
    def validate_targets(cls, target_settings):
        """Turns the given list of target specifications into config objects."""
        return [TargetConfig(**s) for s in target_settings]


class BayBEConfig(BaseModel):
    """Configuration class for BayBE."""

    project_name: str
    parameters: List[dict]
    objective: dict
    random_seed: int = 1337
    allow_repeated_recommendations: bool = True
    allow_recommending_already_measured: bool = True
    numerical_measurements_must_be_within_tolerance: bool = True

    class Config:
        """Pydantic configuration."""

        extra = "forbid"

    @validator("parameters")
    def validate_parameters(cls, param_specs):
        """Turns the given list of parameters specifications into config objects."""
        return [ParameterConfig(**param) for param in param_specs]

    @validator("objective")
    def validate_objective(cls, objective_specs):
        """Turns the given objective specifications into a config object."""
        return ObjectiveConfig(**objective_specs)


# Allowed options and their default values
allowed_config_options = {
    "project_name": "Unnamed Project",
    "random_seed": 1337,
    "allow_repeated_recommendations": True,
    "allow_recommending_already_measured": True,
    "numerical_measurements_must_be_within_tolerance": True,
}


def parse_config(config: dict) -> Tuple[list, list]:
    """
    Parses a BayBE config dictionary. Also sets default values for various flags and
    options directly in the config variable.

    Parameters
    ----------
    config : dict
        A dictionary containing parameter and target info as well as flags and options
        for the method.

    Returns
    -------
    2-Tuple with lists for parsed parameters and targets. The config parameter could
    also be altered because it is assured that all flags and options are set to default
    values
    """

    if ("objective" not in config.keys()) or ("parameters" not in config.keys()):
        raise AssertionError("Your config must define 'parameters' and 'objective'")

    # Parameters
    params = []
    for param in config["parameters"]:
        params.append(parameters.parse_parameter(param))

    # Objective
    objective = config["objective"]
    mode = objective.get("mode", None)
    if mode == "SINGLE":
        targs_dict = objective.get("targets", [])
        if len(targs_dict) != 1:
            raise ValueError(
                f"Config with objective mode SINGLE must specify exactly one target, "
                f"but specified several or none: {targs_dict}"
            )

        target_dict = targs_dict[0]
        targs = [targets.parse_single_target(target_dict)]
    elif mode == "MULTI_DESIRABILITY":
        raise NotImplementedError("This objective mode is not implemented yet")
    elif mode == "MULTI_PARETO":
        raise NotImplementedError("This objective mode is not implemented yet")
    elif mode == "MULTI_TASK":
        raise NotImplementedError("This objective mode is not implemented yet")
    else:
        raise ValueError(
            f"Objective mode is {mode}, but must be one of {targets.allowed_modes}"
        )

    # Options
    for option, value in allowed_config_options.items():
        config.setdefault(option, value)

    # Check for unknown options
    unrecognized_options = [
        key
        for key in config.keys()
        if key not in (list(allowed_config_options) + ["parameters", "objective"])
    ]
    if len(unrecognized_options) > 0:
        raise AssertionError(
            f"The provided config option(s) '{unrecognized_options}'"
            f" is/are not in the allowed "
            f"options {list(allowed_config_options)}"
        )

    return params, targs
