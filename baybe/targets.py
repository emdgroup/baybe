"""
Functionality for different objectives and target variable types.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import partial
from typing import ClassVar, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Extra, validator

from .utils import check_if_in, geom_mean, isabstract, StrictValidationError
from .utils.boundtransforms import bound_bell, bound_linear, bound_triangular

log = logging.getLogger(__name__)


class Objective(BaseModel, extra=Extra.forbid):
    """Class for managing optimization objectives."""

    mode: Literal["SINGLE", "MULTI", "DESIRABILITY"]
    targets: List[dict]
    weights: Optional[List[float]] = None
    combine_func: Literal["MEAN", "GEOM_MEAN"] = "GEOM_MEAN"

    @validator("targets", always=True)
    def validate_targets(cls, targets, values):
        """
        Validates targets depending on the objective mode.
        """
        mode = values["mode"]
        if (mode == "SINGLE") and (len(targets) != 1):
            raise StrictValidationError(
                "For objective mode 'SINGLE', you must specify exactly one target."
            )
        if (mode == "MULTI") and (len(targets) <= 1):
            raise StrictValidationError(
                "For objective mode 'MULTI', you must specify more than one target."
            )
        if mode == "DESIRABILITY":
            for target in targets:
                if ("bounds" not in target) or (target["bounds"] is None):
                    raise StrictValidationError(
                        "In 'DESIRABILITY' mode for multiple targets, each target must "
                        "have bounds defined."
                    )

        return targets

    @validator("weights", always=True)
    def validate_weights(cls, weights, values):
        """
        Validates target weights.
        """
        n_targets = len(values["targets"])

        # Set default: uniform weights
        if weights is None:
            return [100 / n_targets] * n_targets

        if len(weights) != n_targets:
            raise StrictValidationError(
                f"Weights list for your objective has {len(weights)} values, but you "
                f"defined {n_targets} targets."
            )

        # Normalize to sum = 100
        weights = (100 * np.asarray(weights) / np.sum(weights)).tolist()

        return weights

    def transform(self, data: pd.DataFrame, targets: List[Target]) -> pd.DataFrame:
        """
        Transforms targets from experimental to computational representation.

        Parameters
        ----------
        data : pd.DataFrame
            The data to be transformed. Must contain all target values, can contain more
             columns.
        targets : List[Target]
            A list of BayBE targets.

        Returns
        -------
        pd.DataFrame
            A dataframe with the targets in computational representation. Columns will
            be as in the input (except when objective mode is 'DESIRABILITY').
        """
        # Perform transformations that are required independent of the mode
        transformed = data[[t.name for t in targets]].copy()
        for target in targets:
            transformed[target.name] = target.transform(data[target.name])

        # In desirability mode, the targets are additionally combined further into one
        if self.mode == "DESIRABILITY":
            if self.combine_func == "GEOM_MEAN":
                func = geom_mean
            elif self.combine_func == "MEAN":
                func = partial(np.average, axis=1)
            else:
                raise StrictValidationError(
                    f"The specified averaging function {self.combine_func} is unknown."
                )

            vals = func(transformed.values, weights=self.weights)
            transformed = pd.DataFrame({"Comp_Target": vals}, index=transformed.index)

        return transformed


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
        """Registers new subclasses dynamically."""
        super().__init_subclass__(**kwargs)
        if not isabstract(cls):
            cls.SUBCLASSES[cls.type] = cls

    @classmethod
    def create(cls, config: dict) -> Target:
        """Creates a new object matching the given specifications."""
        config = config.copy()
        param_type = config.pop("type")
        check_if_in(param_type, list(Target.SUBCLASSES.keys()))
        return cls.SUBCLASSES[param_type](**config)

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms data into computational representation. The transformation depends
        on the target mode, e.g. minimization, maximization, matching, etc.

        Parameters
        ----------
        data : pd.DataFrame
            The data to be transformed.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the transformed data.
        """


class NumericalTarget(Target):
    """
    Class for numerical targets.
    """

    type = "NUM"

    mode: Literal["MIN", "MAX", "MATCH"]
    bounds: Optional[Tuple[float, float]] = None
    bounds_transform_func: Optional[str] = None

    @validator("bounds", always=True)
    def validate_bounds(cls, bounds, values):
        """
        Currently, either no bounds (= set to None) or completely finite bounds
        (= set to a list of two finite floats) are supported.
        """
        # IMPROVE could also include half-way bounds, which however don't work for the
        #  desirability approach

        if bounds is None:
            if values["mode"] == "MATCH":
                raise StrictValidationError(
                    f"Target '{values['name']}' is in 'MATCH' mode but no bounds were "
                    f"provided. Bounds for 'MATCH' mode are mandatory."
                )
            return None

        if (not isinstance(bounds, tuple)) or (len(bounds) != 2):
            raise StrictValidationError(
                f"Bounds were '{bounds}' but must be a 2-tuple."
            )

        if not all(np.isfinite(bounds)):
            raise StrictValidationError(
                f"Bounds were '{bounds}' but need to contain finite float numbers. "
                f"If you want no bounds, set bounds to 'None'."
            )

        if bounds[1] <= bounds[0]:
            raise StrictValidationError(
                f"The upper bound must be greater than the lower bound. Encountered "
                f"for bounds '{bounds}'."
            )

        return bounds

    @validator("bounds_transform_func", always=True)
    def validate_bounds_transform_func(cls, fun, values):
        """Validates that the given transform is compatible with the specified mode."""

        # Get validated values
        name = values["name"]
        mode = values["mode"]

        # TODO: potentially introduce an abstract base class for the transforms
        #   -> this would remove the necessity to maintain the following dict
        valid_transforms = {
            "MAX": ["LINEAR"],
            "MIN": ["LINEAR"],
            "MATCH": ["TRIANGULAR", "BELL"],
        }

        # Set a default transform
        if (values["bounds"] is not None) and (fun is None):
            fun = valid_transforms[mode][0]
            log.warning(
                "The bound transform function for target '%s' in mode '%s' has not "
                "been specified. Setting the bound transform function to '%s'.",
                name,
                mode,
                fun,
            )

        # Assert that the given transform is valid for the specified target mode
        elif (fun is not None) and (fun not in valid_transforms[mode]):
            raise StrictValidationError(
                f"You specified bounds for target '{name}', but your specified bound "
                f"transform function '{fun}' is not compatible with the target mode "
                f"'{mode}'. It must be one of {valid_transforms[mode]}."
            )

        return fun

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """See base class."""

        transformed = data.copy()

        # TODO: potentially introduce an abstract base class for the transforms
        #   -> this would remove the necessity to maintain the following dict
        #   -> also, it would create a common signature, avoiding the `partial` calls

        # Specify all bound transforms
        bounds_transform_funcs = {
            "LINEAR": bound_linear,
            "TRIANGULAR": bound_triangular,
            "BELL": bound_bell,
        }

        # When bounds are given, apply the respective transform
        if self.bounds is not None:
            func = bounds_transform_funcs[self.bounds_transform_func]
            if self.mode == "MAX":
                func = partial(func, descending=False)
            elif self.mode == "MIN":
                func = partial(func, descending=True)
            transformed = func(transformed, *self.bounds)

        # If no bounds are given, simply negate all target values for "MIN" mode.
        # For "MAX" mode, nothing needs to be done.
        # For "MATCH" mode, the validators avoid a situation without specified bounds.
        elif self.mode == "MIN":
            transformed = -transformed

        return transformed
