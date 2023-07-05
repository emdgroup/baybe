# pylint: disable=missing-function-docstring

"""
Functionality for different objectives and target variable types.
"""
# TODO: ForwardRefs via __future__ annotations are currently disabled due to this issue:
#  https://github.com/python-attrs/cattrs/issues/354

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from attr.validators import instance_of
from attrs import define, field
from attrs.validators import deep_iterable, in_, min_len

from .utils import geom_mean
from .utils.boundtransforms import bound_bell, bound_linear, bound_triangular

from .utils.interval import convert_bounds, Interval
from .utils.serialization import SerialMixin

log = logging.getLogger(__name__)


# TODO: potentially introduce an abstract base class for the transforms
#   -> this would remove the necessity to maintain the following dict
VALID_TRANSFORMS = {
    "MAX": ["LINEAR"],
    "MIN": ["LINEAR"],
    "MATCH": ["TRIANGULAR", "BELL"],
}


@define
class Target(ABC):
    """
    Abstract base class for all target variables. Stores information about the
    range, transformations, etc.
    """

    name: str

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


@define
class NumericalTarget(Target, SerialMixin):
    """
    Class for numerical targets.
    """

    # TODO: Introduce mode enum

    # NOTE: The type annotations of `bounds` are correctly overridden by the attrs
    #   converter. Nonetheless, PyCharm's linter might incorrectly raise a type warning
    #   when calling the constructor. This is a known issue:
    #       https://youtrack.jetbrains.com/issue/PY-34243
    #   Quote from attrs docs:
    #       If a converter’s first argument has a type annotation, that type will
    #       appear in the signature for __init__. A converter will override an explicit
    #       type annotation or type argument.

    mode: Literal["MIN", "MAX", "MATCH"] = field()
    bounds: Interval = field(default=None, converter=convert_bounds)
    bounds_transform_func: Optional[str] = field()

    @bounds_transform_func.default
    def default_bounds_transform_func(self) -> Optional[str]:
        if self.bounds.is_bounded:
            fun = VALID_TRANSFORMS[self.mode][0]
            log.warning(
                "The bound transform function for target '%s' in mode '%s' has not "
                "been specified. Setting the bound transform function to '%s'.",
                self.name,
                self.mode,
                fun,
            )
            return fun
        return None

    @bounds.validator
    def validate_bounds(self, _, value: Interval):
        # Currently, either no bounds or completely finite bounds are supported.
        # IMPROVE: We could also include half-way bounds, which however don't work
        #  for the desirability approach
        if not (value.is_finite or not value.is_bounded):
            raise ValueError("Bounds must either be finite or infinite on *both* ends.")

        if self.mode == "MATCH" and not value.is_finite:
            raise ValueError(
                f"Target '{self.name}' is in 'MATCH' mode, which requires "
                f"finite bounds."
            )

    @bounds_transform_func.validator
    def validate_bounds_transform_func(self, _, value):
        """Validates that the given transform is compatible with the specified mode."""

        # Assert that the given transform is valid for the specified target mode
        if (value is not None) and (value not in VALID_TRANSFORMS[self.mode]):
            raise ValueError(
                f"You specified bounds for target '{self.name}', but your "
                f"specified bound transform function '{value}' is not compatible "
                f"with the target mode {self.mode}'. It must be one "
                f"of {VALID_TRANSFORMS[self.mode]}."
            )

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
        if self.bounds.is_bounded:
            func = bounds_transform_funcs[self.bounds_transform_func]
            if self.mode == "MAX":
                func = partial(func, descending=False)
            elif self.mode == "MIN":
                func = partial(func, descending=True)
            transformed = func(transformed, *self.bounds.to_tuple())

        # If no bounds are given, simply negate all target values for "MIN" mode.
        # For "MAX" mode, nothing needs to be done.
        # For "MATCH" mode, the validators avoid a situation without specified bounds.
        elif self.mode == "MIN":
            transformed = -transformed

        return transformed


@define
class Objective(SerialMixin):
    """Class for managing optimization objectives."""

    # TODO: The class currently directly depends on `NumericalTarget`. Once this
    #   direct dependence is replaced with a dependence on `Target`, the type
    #   annotations should be changed.

    mode: Literal["SINGLE", "DESIRABILITY"]
    targets: List[NumericalTarget] = field(validator=min_len(1))
    weights: List[float] = field(default=None)
    combine_func: Literal["MEAN", "GEOM_MEAN"] = field(
        default="GEOM_MEAN", validator=in_(["MEAN", "GEOM_MEAN"])
    )

    def __attrs_post_init__(self):
        # Use default weights if not provided
        if self.weights is None:
            self.weights = [100] * len(self.targets)

        # Normalize the weights
        self.weights = (100 * np.asarray(self.weights) / np.sum(self.weights)).tolist()

    @targets.validator
    def validate_targets(self, _, targets: List[NumericalTarget]):
        """
        Validates (and instantiates) targets depending on the objective mode.
        """

        # Validate the target specification
        if (self.mode == "SINGLE") and (len(targets) != 1):
            raise ValueError(
                "For objective mode 'SINGLE', exactly one target must be specified."
            )
        if self.mode == "DESIRABILITY":
            if any(not target.bounds.is_bounded for target in targets):
                raise ValueError(
                    "In 'DESIRABILITY' mode for multiple targets, each target must "
                    "have bounds defined."
                )

    @weights.validator
    def validate_weights(self, _, weights):
        """
        Validates target weights.
        """
        if weights is None:
            return

        # Assert that weights is a list of numbers
        validator = deep_iterable(instance_of((int, float)), instance_of(list))
        validator(self, _, weights)

        # Assert that weights has the correct length
        if len(weights) != len(self.targets):
            raise ValueError(
                f"Weights list for your objective has {len(weights)} values, but you "
                f"defined {len(self.targets)} targets."
            )

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms targets from experimental to computational representation.

        Parameters
        ----------
        data : pd.DataFrame
            The data to be transformed. Must contain all target values, can contain
            more columns.

        Returns
        -------
        pd.DataFrame
            A dataframe with the targets in computational representation. Columns will
            be as in the input (except when objective mode is 'DESIRABILITY').
        """
        # Perform transformations that are required independent of the mode
        transformed = data[[t.name for t in self.targets]].copy()
        for target in self.targets:
            transformed[target.name] = target.transform(data[target.name])

        # In desirability mode, the targets are additionally combined further into one
        if self.mode == "DESIRABILITY":
            if self.combine_func == "GEOM_MEAN":
                func = geom_mean
            elif self.combine_func == "MEAN":
                func = partial(np.average, axis=1)
            else:
                raise ValueError(
                    f"The specified averaging function {self.combine_func} is unknown."
                )

            vals = func(transformed.values, weights=self.weights)
            transformed = pd.DataFrame({"Comp_Target": vals}, index=transformed.index)

        return transformed
