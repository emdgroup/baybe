"""
Functionality for different type of targets
"""

from __future__ import annotations

from abc import ABC
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, validator


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

        # TODO Make sure bounds is a 2-sized vector of finite values
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
        if self.mode == "Max":
            if self.bounds is not None and np.isfinite(self.bounds).all():
                # TODO implement transform wth bounds here
                return data * 3
            return data
        if self.mode == "Min":
            if self.bounds is not None and np.isfinite(self.bounds).all():
                # TODO implement transform wth bounds here
                return -data * 3
            return -data
        if self.mode == "Match":
            if self.bounds is not None and np.isfinite(self.bounds).all():
                # TODO implement match transform here
                raise TypeError(
                    f"Match mode is not supported for this target named {self.name} of "
                    f"type {self.TYPE} since it has non-finite bounds or bounds are not"
                    f" defined. Bounds need to be a finite 2-touple."
                )

            raise NotImplementedError("Match mode for targets is not implemented yet.")

        raise ValueError(
            f"The mode '{self.mode}' set for target {self.name} is not recognized. "
            f"Must be either Min, Max or Match"
        )
