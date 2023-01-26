"""
Core functionality of BayBE. Main point of interaction via Python.
"""
from __future__ import annotations

import logging
import pickle
import random
from typing import List, Optional

import fsspec
import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, conlist, Extra, validator

from .constraints import Constraint
from .parameters import Parameter
from .searchspace import SearchSpace
from .strategy import Strategy
from .targets import Objective, Target
from .utils import check_if_in

log = logging.getLogger(__name__)


class NotEnoughPointsLeftError(Exception):
    """An exception raised when more recommendations are requested than there are
    viable parameter configurations left in the search space."""


class BayBEConfig(BaseModel, extra=Extra.forbid):
    """Configuration class for BayBE."""

    # TODO: Remove explicit config class when having found a way to blend the parsing
    #   logic directly into BayBE. Currently, the problem is that additional members
    #   (that do not need to be parsed) cannot be easily defined. This will be fixed
    #   in pydantic 2.0.
    #   - https://github.com/samuelcolvin/pydantic/issues/691
    #   - https://github.com/samuelcolvin/pydantic/issues/1729

    project_name: str = "Untitled Project"
    parameters: conlist(dict, min_items=1)
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

    def __init__(self, config: BayBEConfig, searchspace: Optional[SearchSpace] = None):
        """
        Constructor of the BayBE class.

        Parameters
        ----------
        config : BayBEConfig
            Pydantic-validated config object.
        searchspace : SearchSpace (optional)
            An optional search space object. If provided, the search space will not
            be created from the config but instead the given object is used.
        """
        # Set global random seeds
        torch.manual_seed(config.random_seed)
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Store the configuration
        self.config = config

        # Current iteration/batch number
        self.batches_done: int = 0

        # Flag to indicate if the specified recommendation strategy is "random", in
        # which case certain operation can be skipped, such as the (potentially
        # costly) transformation of the parameters into computation representation.
        self._random: bool = config.strategy.get("recommender_cls", "") == "RANDOM"

        # Initialize all subcomponents
        if searchspace is None:
            parameters = [Parameter.create(p) for p in config.parameters]
            constraints = [Constraint.create(c) for c in config.constraints]
            self.searchspace = SearchSpace(parameters, constraints, self._random)
        else:
            self.searchspace = searchspace
        self.objective = Objective(**config.objective)
        self.strategy = Strategy(**config.strategy, searchspace=self.searchspace)

        # Declare variable for storing measurements (in experimental representation)
        self.measurements_exp = None

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
        return self.searchspace.transform(self.measurements_exp)

    @property
    def measurements_targets_comp(self) -> pd.DataFrame:
        """The computational representation of the measured targets."""
        return self.objective.transform(self.measurements_exp)

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

        string += "\n\nSearch Space (Discrete Part, Experimental Representation):\n"
        string += f"{self.searchspace.discrete.exp_rep}"

        string += "\n\nSearch Space (Discrete Part, Computational Representation):\n"
        string += f"{self.searchspace.discrete.comp_rep}"

        string += "\n\nSearch Space (Continuous Part, Boundaries):\n"
        string += f"{self.searchspace.continuous.bounds}"

        string += "\n\nMeasurement Space (Experimental Representation):\n"
        string += f"{self.measurements_exp}"

        string += "\n\nMeasurement Space (Computational Representation):\n"
        string += f"{self.measurements_parameters_comp}\n"
        string += f"{self.measurements_targets_comp}"

        return string

    def state_dict(self) -> dict:
        """Creates a dictionary representing the object's internal state."""
        state_dict = dict(
            config_dict=self.config.dict(),
            batches_done=self.batches_done,
            searchspace=self.searchspace.state_dict(),
            measurements=self.measurements_exp,
        )
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Restores a given object state."""

        # Overwrite the member variables with the given state information
        self.config = state_dict["config"]
        self.batches_done = state_dict["batches_done"]
        self.measurements_exp = state_dict["measurements"]

        # Restore the search space state
        # TODO: Extend the load_state_dict function of SearchSpace such that it takes
        #   care of everything. For that, we need state_dict functionality for all
        #   BayBE components.
        self.searchspace = SearchSpace(self.parameters, self.constraints, self._random)
        self.searchspace.load_state_dict(state_dict["searchspace"])

        # Restore the strategy state
        # TODO: implement state_dict functionality for Strategy
        self.strategy = Strategy(
            searchspace=self.searchspace,
            **self.config.strategy,
        )
        self.strategy.fit(
            self.measurements_parameters_comp, self.measurements_targets_comp
        )

    @classmethod
    def from_file(cls, path: str, **kwargs) -> BayBE:
        """
        Class method to restore a BayBE object that has been saved to disk.

        Parameters
        ----------
        path : str
            Path to the stored BayBE object.
        kwargs : keyword arguments
            Additional arguments passed to fsspec.open. Useful, for instance, for
            accessing remote or s3 file systems.

        Returns
        -------
        BayBE
            The restored BayBE instance.
        """
        # Load stored BayBE state
        with fsspec.open(path, **kwargs) as file:
            state_dict = pickle.load(file)

        # Parse the stored configuration file
        config = BayBEConfig(**state_dict["config_dict"])

        # To avoid creating the search space from scratch and reduce the computational
        # effort to build the associated parameter representations, create a "blank"
        # search space object and restore its state from the file.
        # TODO: It is not ensured that the given parameters/constraints are consistent
        #   with the provided search space state coming from the state dict. A proper
        #   validation would require that the searchspace state dict also carries
        #   the underlying parameter/constraint information, which could then be
        #   compared with the given specifications.
        parameters = [Parameter.create(p) for p in config.parameters]
        constraints = [Constraint.create(c) for c in config.constraints]
        searchspace = SearchSpace(parameters, constraints, init_dataframes=False)
        searchspace.load_state_dict(state_dict["searchspace"])

        # Create the BayBE object using the stored config and pre-initialized search
        # space via the constructor
        baybe = cls(config, searchspace)

        # Restore its state
        state_dict["config"] = config
        baybe.load_state_dict(state_dict)

        return baybe

    def save(self, path: Optional[str] = None, **kwargs) -> None:
        """
        Store the current state of the BayBE instance on disk.

        Parameters
        ----------
        path : str
            Path to where the BayBE object should be stored.
        kwargs : keyword arguments
            Additional arguments passed to fsspec.open. Useful, for instance, for
            accessing remote or s3 file systems.

        Returns
        -------
        Nothing.
        """
        # If no path is provided, use a default file path
        if path is None:
            path = "./baybe_object.baybe"
            log.warning(
                "No path was specified for storing the BayBE object. Will use '%s'.",
                path,
            )

        # Write the BayBE state to disk
        with fsspec.open(path, "wb", **kwargs) as file:
            pickle.dump(self.state_dict(), file)

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
            if (param.is_numeric) and (data[param.name].dtype.kind not in "iufb"):
                raise TypeError(
                    f"The numerical parameter '{param.name}' has non-numeric entries in"
                    f" the provided dataframe."
                )

        # Update meta data
        self.searchspace.discrete.mark_as_measured(
            data, self.config.numerical_measurements_must_be_within_tolerance
        )

        # Read in measurements and add them to the database
        self.batches_done += 1
        to_insert = data.copy()
        to_insert["BatchNr"] = self.batches_done

        self.measurements_exp = pd.concat(
            [self.measurements_exp, to_insert], axis=0, ignore_index=True
        )

    def recommend(self, batch_quantity: int = 5) -> pd.DataFrame:
        """
        Provides the recommendations for the next batch of experiments.
        """
        candidates_exp, candidates_comp = self.searchspace.discrete.get_candidates(
            self.config.allow_repeated_recommendations,
            self.config.allow_recommending_already_measured,
        )

        # Assert that there are enough points left for recommendation
        if len(candidates_exp) < batch_quantity:
            raise NotEnoughPointsLeftError(
                f"Using the current settings, there are fewer than {batch_quantity} "
                "possible data points left to recommend. This can be "
                "either because all data points have been measured at some point "
                "(while 'allow_repeated_recommendations' or "
                "'allow_recommending_already_measured' being False) "
                "or because all data points are marked as 'dont_recommend'."
            )

        # Update the strategy object
        self.strategy.fit(
            self.measurements_parameters_comp, self.measurements_targets_comp
        )

        # Get the indices of the recommended search space entries
        idxs = self.strategy.recommend(candidates_comp, batch_quantity=batch_quantity)

        # Translate indices into labeled data points and update metadata
        # TODO: Don't modify searchspace members directly. Probably, the metadata
        #   should become part of the BayBE class, which would cleanly separate
        #   responsibilities. That is, BayBE would capture all data-related information,
        #   reflecting the progress of an experiment, whereas the SearchSpace class
        #   would be a stateless representation of the mathematical search space.
        rec = candidates_exp.loc[idxs, :]
        self.searchspace.discrete.metadata.loc[idxs, "was_recommended"] = True

        # Query user input
        for target in self.targets:
            rec[target.name] = "<Enter value>"

        return rec
