# pylint: disable = no-self-use, no-self-argument
"""
Config functionality
"""

import logging
from typing import Any, List, Optional

from pydantic import BaseModel, validator

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
    tolerance: Optional[float]  # TODO: conditional validation depending on type
    encoding: Optional[str]  # TODO: conditional validation depending on type

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
