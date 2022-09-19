"""
Core functionality of BayBE. Main point of interaction via Python.
"""
import logging
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from pydantic import BaseModel, Extra, validator

from . import parameters as baybe_parameters
from .parameters import Parameter
from .strategy import Strategy
from .targets import Objective, Target
from .utils import check_if_in

log = logging.getLogger(__name__)


class BayBEConfig(BaseModel, extra=Extra.forbid):
    """Configuration class for BayBE."""

    # TODO: Remove explicit config class when having found a way to blend the parsing
    #   logic directly into BayBE. Currently, the problem is that additional members
    #   (that do not need to be parsed) cannot be easily defined. This will be fixed
    #   in pydantic 2.0.
    #   - https://github.com/samuelcolvin/pydantic/issues/691
    #   - https://github.com/samuelcolvin/pydantic/issues/1729

    project_name: str = "Untitled Project"
    parameters: List[dict]
    objective: dict
    strategy: Optional[dict] = None
    random_seed: int = 1337
    allow_repeated_recommendations: bool = True
    allow_recommending_already_measured: bool = True
    numerical_measurements_must_be_within_tolerance: bool = True

    @validator("parameters")
    def validate_parameter_types(cls, param_specs):
        """
        Validates that each parameter has a valid type.
        All remaining parameter specifications are validated during instantiation.
        """
        try:
            for param in param_specs:
                check_if_in(param["type"], list(Parameter.SUBCLASSES.keys()))
        except KeyError as exc:
            raise ValueError(
                "Each parameter needs a valid type specification."
            ) from exc
        return param_specs

    @validator("strategy", always=True)
    def validate_strategy(cls, strategy):
        """Sets the default strategy options to the empty set."""
        return strategy or {}


class BayBE:
    """Main class for interaction with BayBE."""

    def __init__(self, config: BayBEConfig):
        # Set global random seeds
        torch.manual_seed(config.random_seed)
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Current iteration/batch number
        self.batches_done = 0

        # Parse config and create all model components except the strategy (which
        # currently needs the computational representation of the search space)
        # TODO: derive the required information directly from the Parameter objects
        self.config = config
        self.parameters = [Parameter.create(p) for p in config.parameters]
        self.objective = Objective(**config.objective)
        self.targets = [Target.create(t) for t in self.objective.targets]

        # Create a dataframe representing the experimental search space
        self.searchspace_exp_rep = baybe_parameters.parameter_cartesian_prod_to_df(
            self.parameters
        )
        self.searchspace_metadata = pd.DataFrame(
            {
                "was_recommended": False,
                "was_measured": False,
                "dont_recommend": False,
            },
            index=self.searchspace_exp_rep.index,
        )

        # Create a corresponding dataframe containing the computational representation
        self.searchspace_comp_rep, _ = self.transform_rep_exp2comp(
            self.searchspace_exp_rep
        )

        # Declare measurement dataframes
        self.measurements_exp_rep = None
        self.measurements_comp_rep_x = None
        self.measurements_comp_rep_y = None

        # Initialize the DOE strategy
        self.strategy = Strategy(
            **config.strategy, searchspace=self.searchspace_comp_rep
        )

    def transform_rep_exp2comp(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Transforms a dataframe from experimental to computational representation.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be transformed. Must contain all parameter columns. Can additionally
            contain all target columns, which get transformed separately.

        Returns
        -------
        Tuple[pd.DataFrame, Optional[pd.DataFrame]]
            Transformed parameters and, if contained in the input, transformed targets.
        """
        # Transform the parameters
        dfs = []
        for param in self.parameters:
            comp_df = param.transform_rep_exp2comp(data[param.name])
            dfs.append(comp_df)
        comp_rep_x = pd.concat(dfs, axis=1)

        # Transform the (optional) targets
        comp_rep_y = None
        if all(target.name in data.columns for target in self.targets):
            comp_rep_y = self.objective.transform(data=data, targets=self.targets)

        return comp_rep_x, comp_rep_y

    def __str__(self):
        """
        Prints a simple summary of the BayBE object. Some information provided here
        might not be relevant for production-ready code.
        """

        string = "\nTarget and Parameters:\n"
        for target in self.targets:
            string += f"{target}\n"
        for param in self.parameters:
            string += f"{param}\n"

        string += "Options:\n"
        for option, value in self.config.dict().items():
            string += f"   {option}: {value}\n"

        string += "\n\nSearch Space (Experimental Representation):\n"
        string += f"{self.searchspace_exp_rep}"

        string += "\n\nSearch Space (Computational Representation):\n"
        string += f"{self.searchspace_comp_rep}"

        string += "\n\nMeasurement Space (Experimental Representation):\n"
        string += f"{self.measurements_exp_rep}"

        string += "\n\nMeasurement Space (Computational Representation):\n"
        string += f"{self.measurements_comp_rep_x}\n"
        string += f"{self.measurements_comp_rep_y}"

        return string

    def _match_measurement_with_searchspace_indices(
        self, data: pd.DataFrame
    ) -> pd.Index:
        """
        Matches rows of a dataframe (e.g. measurements to be added to the internal data)
        to the indices of the search space dataframe. This is useful for validity checks
        and to automatically match measurements to entries in the search space, e.g. to
        detect which ones have been measured. For categorical parameters, there needs
        to be an exact match with any of the allowed values. For numerical parameters,
        the user can decide via a BayBE flag whether values outside the tolerance should
        be accepted.

        Parameters
        ----------
        data : pd.DataFrame
            The data that should be checked for matching entries in the search space.

        Returns
        -------
        pd.Index
            The index of the matching search space entries.
        """
        # IMPROVE: neater implementation (e.g. via fuzzy join)

        inds_matched = []

        # Iterate over all input rows
        for ind, row in data.iterrows():

            # Check if the row represents a valid input
            valid = True
            for param in self.parameters:
                if "NUM" in param.type:
                    if self.config.numerical_measurements_must_be_within_tolerance:
                        valid &= param.is_in_range(row[param.name])
                else:
                    valid &= param.is_in_range(row[param.name])
                if not valid:
                    raise ValueError(
                        f"Input data on row with the index {row.name} has invalid "
                        f"values in parameter '{param.name}'. "
                        f"For categorical parameters, values need to exactly match a "
                        f"valid choice defined in your config. "
                        f"For numerical parameters, a match is accepted only if "
                        f"the input value is within the specified tolerance/range. Set "
                        f"the flag 'numerical_measurements_must_be_within_tolerance' "
                        f"to 'False' to disable this behavior."
                    )

            # Identify discrete and numeric parameters
            # TODO: This is error-prone. Add member to indicate discreteness or
            #  introduce corresponding super-classes.
            cat_cols = [
                param.name for param in self.parameters if "NUM" not in param.type
            ]
            num_cols = [param.name for param in self.parameters if "NUM" in param.type]

            # Discrete parameters must match exactly
            match = (
                self.searchspace_exp_rep[cat_cols]
                .eq(row[cat_cols])
                .all(axis=1, skipna=False)
            )

            # For numeric parameters, match the entry with the smallest deviation
            # TODO: allow alternative distance metrics
            for param in num_cols:
                abs_diff = (self.searchspace_exp_rep[param] - row[param]).abs()
                match &= abs_diff == abs_diff.min()

            # We expect exactly one match. If that's not the case, print a warning.
            inds_found = self.searchspace_exp_rep.index[match].to_list()
            if len(inds_found) == 0:
                log.warning(
                    "Input row with index %s could not be matched to the search space. "
                    "This could indicate that something went wrong.",
                    ind,
                )
            elif len(inds_found) > 1:
                log.warning(
                    "Input row with index %s has multiple matches with "
                    "the search space. This could indicate that something went wrong. "
                    "Matching only first occurrence.",
                    ind,
                )
                inds_matched.append(inds_found[0])
            else:
                inds_matched.extend(inds_found)

        return pd.Index(inds_matched)

    def add_results(self, data: pd.DataFrame) -> None:
        """
        Adds results from a dataframe to the internal database and updates the strategy
        object accordingly.

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
        # Check if all provided data points have acceptable parameter values
        inds_matched = self._match_measurement_with_searchspace_indices(data)

        # Check if all targets have values provided
        for target in self.targets:
            if data[target.name].isna().any():
                raise ValueError(
                    f"The target '{target.name}' has missing values or NaNs in the "
                    f"provided dataframe. Missing target values are not currently "
                    f"supported."
                )

        # Update the 'was_measured' metadata
        self.searchspace_metadata.loc[inds_matched, "was_measured"] = True

        # Read in measurements and add them to the database
        self.batches_done += 1
        data["BatchNr"] = self.batches_done
        self.measurements_exp_rep = pd.concat(
            [self.measurements_exp_rep, data], axis=0, ignore_index=True
        )

        # Transform measurement space to computational representation
        (
            self.measurements_comp_rep_x,
            self.measurements_comp_rep_y,
        ) = self.transform_rep_exp2comp(self.measurements_exp_rep)

        # Update the strategy object
        self.strategy.fit(self.measurements_comp_rep_x, self.measurements_comp_rep_y)

    def recommend(self, batch_quantity: int = 5) -> pd.DataFrame:
        """
        Provides the recommendations for the next batch of experiments.
        """
        # Filter the search space before passing it to the strategy
        mask_todrop = self.searchspace_metadata["dont_recommend"].copy()
        if not self.config.allow_repeated_recommendations:
            mask_todrop |= self.searchspace_metadata["was_recommended"]
        if not self.config.allow_recommending_already_measured:
            mask_todrop |= self.searchspace_metadata["was_measured"]

        # Assert that there are enough points left for recommendation
        # TODO: use available of points left and show a warning
        if (mask_todrop.sum() >= len(self.searchspace_exp_rep)) or (
            len(self.searchspace_exp_rep.loc[~mask_todrop]) < batch_quantity
        ):
            raise AssertionError(
                f"Using the current settings, there are fewer than '{batch_quantity=}' "
                f"possible data points left to recommend. This can be either because "
                f"all data points have been measured at some point (while "
                f"'allow_repeated_recommendations' or "
                "'allow_recommending_already_measured' being False) or because all "
                "data points are marked as 'dont_recommend'."
            )

        # Get the indices of the recommended search space entries
        inds = self.strategy.recommend(
            self.searchspace_comp_rep.loc[~mask_todrop], batch_quantity=batch_quantity
        )

        # Translate indices into labeled data points and update metadata
        rec = self.searchspace_exp_rep.loc[inds, :]
        self.searchspace_metadata.loc[inds, "was_recommended"] = True

        # Query user input
        for target in self.targets:
            rec[target.name] = "<Enter value>"

        return rec

    def load(self) -> None:
        """
        Load new internal state of a DOE from a specified file
        The load and save functions could also be omitted and the user would have to
        take care of simply storing the BayBE object eg via dill. This could
        potentially create problems when code versions are different
        """
        # TODO: Implement and revise docstring + type hints accordingly
        raise NotImplementedError("Loading a BayBE object is not implemented yet")

    def save(self) -> None:
        """
        Store the current state of the DOE on disk
        """
        # TODO: Implement and revise docstring + type hints accordingly
        raise NotImplementedError("Saving a BayBE object is not implemented yet")
