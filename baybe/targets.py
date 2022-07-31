"""
Functionality for different type of targets
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Extra, validator

from baybe.utils import check_if_in


class Objective(BaseModel, extra=Extra.forbid):
    """Class for managing optimization objectives."""

    mode: Literal["SINGLE"]
    targets: List[dict]

    @validator("targets")
    def validate_targets(cls, targets):
        """Validates that only one target has been specified."""
        if len(targets) > 1:
            raise ValueError("Currently, only one target is supported.")
        return targets


class Target(ABC, BaseModel, extra=Extra.forbid):
    """
    Abstract base class for all target variables. Stores information about the type,
    range, transformations, etc.
    """

    # class variables
    type: ClassVar[str]
    SUBCLASSES: ClassVar[Dict[str, Target]] = {}

    # object variables
    name: str

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.type] = cls

    @classmethod
    def create(cls, config: dict) -> Target:
        """Creates a new target object matching the given specifications."""
        config = config.copy()
        param_type = config.pop("type")
        check_if_in(param_type, list(Target.SUBCLASSES.keys()))
        return cls.SUBCLASSES[param_type](**config)


class NumericalTarget(Target):
    """
    Class for numerical targets
    """

    type = "NUM"

    mode: Literal["MIN", "MAX", "MATCH"]
    bounds: Optional[Tuple[float, float]]

    @validator("bounds")
    def validate_bounds(cls, bounds):
        """Validates if the given bounds are specified correctly."""
        if bounds[1] <= bounds[0]:
            raise ValueError("The upper bound must be greater than the lower bound.")
        return bounds

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms data to the computational representation. The transformation depends
        on the target mode, e.g. minimization, maximization, matching, multi-target etc.

        Parameters
        ----------
        data : pd.DataFrame
            The data to be transformed.

        Returns
        -------
        pd.DataFrame
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
                    f"type {self.type} since it has non-finite bounds or bounds are not"
                    f" defined. Bounds need to be a finite 2-tuple."
                )
            raise NotImplementedError("MATCH mode for targets is not implemented yet.")
        return data
