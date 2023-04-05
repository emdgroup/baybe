"""
Core functionality of BayBE. Main point of interaction via Python.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from pydantic import Field, parse_obj_as

from baybe.constraints import Constraint
from baybe.parameters import Parameter
from baybe.searchspace import SearchSpace
from baybe.strategies.strategy import Strategy
from baybe.targets import Objective, Target
from baybe.utils import BaseModel

log = logging.getLogger(__name__)


class BayBE(BaseModel):
    """Main class for interaction with BayBE."""

    # DOE specifications
    searchspace: SearchSpace
    objective: Objective
    strategy: Strategy

    # Data
    measurements_exp: pd.DataFrame = Field(default_factory=pd.DataFrame)
    numerical_measurements_must_be_within_tolerance: bool = True

    # Metadata
    batches_done: int = 0
    fits_done: int = 0

    # TODO: make private
    cached_recommendation: pd.DataFrame = Field(default_factory=pd.DataFrame)

    @property
    def parameters(self) -> List[Parameter]:
        """The parameters of the underlying search space."""
        return self.searchspace.parameters

    @property
    def constraints(self) -> List[Constraint]:
        """The parameter constraints of the underlying search space."""
        return self.searchspace.constraints

    @property
    def targets(self) -> List[Target]:
        """The targets of the underlying objective."""
        return self.objective.targets

    @property
    def measurements_parameters_comp(self) -> pd.DataFrame:
        """The computational representation of the measured parameters."""
        if len(self.measurements_exp) < 1:
            return pd.DataFrame()
        return self.searchspace.transform(self.measurements_exp)

    @property
    def measurements_targets_comp(self) -> pd.DataFrame:
        """The computational representation of the measured targets."""
        if len(self.measurements_exp) < 1:
            return pd.DataFrame()
        return self.objective.transform(self.measurements_exp)

    @classmethod
    def from_dict(cls, config: dict) -> BayBE:
        """Creates a BayBE object from a config dictionary."""
        # TODO: This is only a temporary workaround that allows all current tests
        #   to pass without having to replace their config dicts. Will be removed
        #   once all config dicts have been ditched.

        # Move the boolean flags to the strategy dict
        if "strategy" not in config:
            config["strategy"] = {}
        flags = {
            k: config.pop(k)
            for k in [
                "allow_repeated_recommendations",
                "allow_recommending_already_measured",
            ]
            if k in config
        }
        for key, val in flags.items():
            config["strategy"][key] = val

        # Drop removed keys
        for key in ["random_seed", "project_name"]:
            config.pop(key, None)

        # Pre-parse parameters and constraints to avoid having to hook into pydantic's
        # parsing process to call Searchspace.create instead of the default constructor
        parameters = parse_obj_as(List[Parameter], config["parameters"])
        if "constraints" in config:
            constraints = parse_obj_as(List[Constraint], config.pop("constraints"))
        else:
            constraints = None
        config["searchspace"] = SearchSpace.create(
            parameters=parameters,
            constraints=constraints,
        )

        return parse_obj_as(BayBE, config)

    def add_results(self, data: pd.DataFrame) -> None:
        """
        Adds results from a dataframe to the internal database.

        Each addition of data is considered a new batch. Added results are checked for
        validity. Categorical values need to have an exact match. For numerical values,
        a BayBE flag determines if values that lie outside a specified tolerance
        are accepted.

        Parameters
        ----------
        data : pd.DataFrame
            The data to be added (with filled values for targets). Preferably created
            via the `recommend` method.

        Returns
        -------
        Nothing (the internal database is modified in-place).
        """
        # Invalidate recommendation cache first (in case of uncaught exceptions below)
        self.cached_recommendation = pd.DataFrame()

        # Check if all targets have valid values
        for target in self.targets:
            if data[target.name].isna().any():
                raise ValueError(
                    f"The target '{target.name}' has missing values or NaNs in the "
                    f"provided dataframe. Missing target values are not supported."
                )
            if data[target.name].dtype.kind not in "iufb":
                raise TypeError(
                    f"The target '{target.name}' has non-numeric entries in the "
                    f"provided dataframe. Non-numeric target values are not supported."
                )

        # Check if all targets have valid values
        for param in self.parameters:
            if data[param.name].isna().any():
                raise ValueError(
                    f"The parameter '{param.name}' has missing values or NaNs in the "
                    f"provided dataframe. Missing parameter values are not supported."
                )
            if param.is_numeric and (data[param.name].dtype.kind not in "iufb"):
                raise TypeError(
                    f"The numerical parameter '{param.name}' has non-numeric entries in"
                    f" the provided dataframe."
                )

        # Update meta data
        # TODO: refactor responsibilities
        self.searchspace.discrete.mark_as_measured(
            data, self.numerical_measurements_must_be_within_tolerance
        )

        # Read in measurements and add them to the database
        # TODO: See if np.nan can be replaced with pd.NA once (de-)serialization is
        #   in place. The current serializer set in the pydantic config does not support
        #   pd.NA. Pandas' .to_json() can handle it but reading it back gives also
        #   np.nan as result. Probably, the only way is a custom (de-)serializer.
        self.batches_done += 1
        to_insert = data.copy()
        to_insert["BatchNr"] = self.batches_done
        to_insert["FitNr"] = np.nan

        self.measurements_exp = pd.concat(
            [self.measurements_exp, to_insert], axis=0, ignore_index=True
        )

    def recommend(self, batch_quantity: int = 5) -> pd.DataFrame:
        """
        Provides the recommendations for the next batch of experiments.

        Parameters
        ----------
        batch_quantity : int > 0
            Number of requested recommendations.

        Returns
        -------
        rec : pd.DataFrame
            Contains the recommendations in experimental representation.
        """
        if batch_quantity < 1:
            raise ValueError(
                f"You must at least request one recommendation per batch, but provided "
                f"{batch_quantity=}."
            )

        # If there are cached recommendations and the batch size of those is equal to
        # the previously requested one, we just return those
        if len(self.cached_recommendation) == batch_quantity:
            return self.cached_recommendation

        # Update recommendation meta data
        if len(self.measurements_exp) > 0:
            self.fits_done += 1
            self.measurements_exp["FitNr"].fillna(self.fits_done, inplace=True)

        # Get the recommended search space entries
        rec = self.strategy.recommend(
            self.searchspace,
            self.measurements_parameters_comp,
            self.measurements_targets_comp,
            batch_quantity,
        )

        # Query user input
        for target in self.targets:
            rec[target.name] = "<Enter value>"

        # Cache the recommendations
        self.cached_recommendation = rec.copy()

        return rec
