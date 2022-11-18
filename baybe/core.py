"""
Core functionality of BayBE. Main point of interaction via Python.
"""
from __future__ import annotations

import logging
import random
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Extra, validator

from .constraints import Constraint
from .parameters import Parameter
from .searchspace import SearchSpace
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
    constraints: List[dict] = []

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

    def __init__(self, config: BayBEConfig, create_searchspace: bool = True):
        """
        Constructor of the BayBE class.

        Parameters
        ----------
        config : BayBEConfig
            Pydantic-validated config object.
        create_searchspace : bool
            Indicator that allows skipping the creation of searchspace and strategy.
            Useful when using the constructor to create a BayBE object from stored data
            (in that case, searchspace is loaded from disk and not created from config).
        """
        # Set global random seeds
        torch.manual_seed(config.random_seed)
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Current iteration/batch number
        self.batches_done = 0

        # Config
        # TODO: derive the required information directly from the Parameter objects
        # TODO: find a better solution for the self._random property hack
        self.config = config
        self._random = (
            config.strategy.get("recommender_cls", "UNRESTRICTED_RANKING") == "RANDOM"
        )

        # Create Parameter, Objective and Constraint objects
        self.parameters = [Parameter.create(p) for p in config.parameters]
        self.objective = Objective(**config.objective)
        self.constraints = [Constraint.create(c) for c in config.constraints]

        if create_searchspace:

            self.searchspace = SearchSpace(
                self.parameters, self.constraints, self._random
            )

            # Initialize the DOE strategy
            self.strategy = Strategy(**config.strategy, searchspace=self.searchspace)

        else:
            self.searchspace = None
            self.strategy = None

        # Declare measurement dataframe
        self.measurements = None

    @property
    def targets(self) -> List[Target]:
        """Returns the targets of the underlying objective."""
        return self.objective.targets

    @property
    def measured_parameters_comp(self) -> pd.DataFrame:
        """Returns the computational representation of the measured parameters."""
        return self.searchspace.transform(self.measurements)

    @property
    def measured_targets_comp(self) -> pd.DataFrame:
        """Returns the computational representation of the measured targets."""
        return self.objective.transform(self.measurements)

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
        string += f"{self.searchspace.exp_rep}"

        string += "\n\nSearch Space (Computational Representation):\n"
        string += f"{self.searchspace.comp_rep}"

        string += "\n\nMeasurement Space (Experimental Representation):\n"
        string += f"{self.measurements}"

        string += "\n\nMeasurement Space (Computational Representation):\n"
        string += f"{self.measured_parameters_comp}\n"
        string += f"{self.measured_targets_comp}"

        return string

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
        # Check if all targets have values provided
        for target in self.targets:
            if data[target.name].isna().any():
                raise ValueError(
                    f"The target '{target.name}' has missing values or NaNs in the "
                    f"provided dataframe. Missing target values are not currently "
                    f"supported."
                )

        self.searchspace.mark_as_measured(
            data, self.config.numerical_measurements_must_be_within_tolerance
        )

        # Read in measurements and add them to the database
        self.batches_done += 1
        to_insert = data.copy()
        to_insert["BatchNr"] = self.batches_done

        self.measurements = pd.concat(
            [self.measurements, to_insert], axis=0, ignore_index=True
        )

        # Update the strategy object
        self.strategy.fit(self.measured_parameters_comp, self.measured_targets_comp)

    def recommend(self, batch_quantity: int = 5) -> pd.DataFrame:
        """
        Provides the recommendations for the next batch of experiments.
        """
        candidates_exp, candidates_comp = self.searchspace.get_candidates(
            self.config.allow_repeated_recommendations,
            self.config.allow_recommending_already_measured,
        )

        # Assert that there are enough points left for recommendation
        if len(candidates_exp) < batch_quantity:
            log.warning(
                "Using the current settings, there are fewer than %s "
                "possible data points left to recommend. This can be "
                "either because all data points have been measured at some point "
                "(while 'allow_repeated_recommendations' or "
                "'allow_recommending_already_measured' being False) "
                "or because all data points are marked as 'dont_recommend'.",
                batch_quantity,
            )

        # Get the indices of the recommended search space entries
        idxs = self.strategy.recommend(candidates_comp, batch_quantity=batch_quantity)

        # Translate indices into labeled data points and update metadata
        rec = candidates_exp.loc[idxs, :]  # TODO: do we need a copy here?
        self.searchspace.metadata.loc[
            idxs, "was_recommended"
        ] = True  # TODO: don't modify searchspace members directly

        # Query user input
        for target in self.targets:
            rec[target.name] = "<Enter value>"

        return rec

    @classmethod
    def from_stored(cls, path: str, **kwargs) -> BayBE:
        """
        Class method to create a BayBE object from a stored object.

        Parameters
        ----------
        path : str
            Path to the stored object.
        kwargs : keyword arguments
            Additional arguments passed to fsspec.open. Useful, for instance, for
            accessing remote or s3 file systems.

        Returns
        -------
        BayBE
            The restored BayBE instance.
        """
        raise NotImplementedError("This method is currently not functional.")

    def save(self, path: Optional[str] = None, **kwargs) -> None:
        """
        Store the current state of the BayBE instance on disk.

        Parameters
        ----------
        path : str
            Path to where the object should be stored.
        kwargs : keyword arguments
            Additional arguments passed to fsspec.open. Useful, for instance, for
            accessing remote or s3 file systems.

        Returns
        -------
        Nothing.
        """
        raise NotImplementedError("This method is currently not functional.")
