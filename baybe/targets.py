"""
Functionality for different type of targets
"""

from __future__ import annotations

from abc import ABC
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Extra, validator

from baybe.utils import check_if_in


class TargetConfig(BaseModel, extra=Extra.forbid):
    """Configuration class for creating target objects."""

    name: str
    type: str
    mode: Literal["MIN", "MAX", "MATCH"]
    bounds: Optional[Tuple[float, float]]

    @validator("type")
    def validate_type(cls, val):
        """Validates if the given target type exists."""
        check_if_in(val, Target.SUBCLASSES)
        return val

    @validator("bounds")
    def validate_bounds(cls, bounds):
        """Validates if the given bounds are specified correctly."""
        if bounds[1] <= bounds[0]:
            raise ValueError("The upper bound must be greater than the lower bound.")
        return bounds


class ObjectiveConfig(BaseModel, extra=Extra.forbid):
    """Configuration class for creating objective objects."""

    mode: str
    targets: List[dict]

    @validator("targets")
    def validate_targets(cls, target_settings):
        """Turns the given list of target specifications into config objects."""
        return [TargetConfig(**s) for s in target_settings]


class Target(ABC):
    """
    Abstract base class for all target variables. Stores information about the type,
    range, transformations, etc.
    """

    TYPE: str
    SUBCLASSES: Dict[str, Target] = {}

    def __init__(self, config: TargetConfig):
        self.name = config.name

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.TYPE] = cls

    @classmethod
    # TODO: add type hint once circular import problem has been fixed
    def create(cls, config) -> Target:
        """Creates a new target object matching the given specifications."""
        return cls.SUBCLASSES[config.type](config)

    @classmethod
    def from_dict(cls, config_dict: dict) -> Target:
        """Creates a target from a config dictionary."""
        return cls(TargetConfig(**config_dict))


class NumericalTarget(Target):
    """
    Class for numerical targets
    """

    TYPE = "NUM"

    def __init__(self, config: TargetConfig):
        super().__init__(config)
        self.mode = config.mode
        self.bounds = config.bounds

    def __str__(self):
        string = (
            f"Numerical target\n"
            f"   Name:   '{self.name}'\n"
            f"   Mode:   '{self.mode}'\n"
            f"   Bounds: {self.bounds}"
        )
        return string

    def transform(self, data: pd.DataFrame):
        """
        Transform data to the computational representation. The transformation depends
        on the target mode, e.g. minimization, maximization, matching, multi-target etc

        Parameters
        ----------
        data: pd.DataFrame
            The data to be transformed.
        Returns
        -------
        The transformed data frame
        """

        # TODO implement transforms for bounds
        if self.mode == "MAX":
            if self.bounds is not None and np.isfinite(self.bounds).all():
                # TODO implement transform wth bounds here
                raise NotImplementedError()
            return data
        if self.mode == "MIN":
            if self.bounds is not None and np.isfinite(self.bounds).all():
                # TODO implement transform wth bounds here
                raise NotImplementedError()
            return -data
        if self.mode == "MATCH":
            if self.bounds is not None and np.isfinite(self.bounds).all():
                # TODO implement match transform here
                raise TypeError(
                    f"MATCH mode is not supported for this target named {self.name} of "
                    f"type {self.TYPE} since it has non-finite bounds or bounds are not"
                    f" defined. Bounds need to be a finite 2-tuple."
                )
            raise NotImplementedError("MATCH mode for targets is not implemented yet.")
        return data
