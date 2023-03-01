"""
Functionality for managing search spaces.
"""
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .constraints import _constraints_order, Constraint
from .parameters import (
    DiscreteParameter,
    NumericContinuous,
    Parameter,
    parameter_cartesian_prod_to_df,
)
from .utils import df_drop_single_value_columns

log = logging.getLogger(__name__)
INF_BOUNDS_REPLACEMENT = 1000


class SearchSpace:
    """
    Class for managing the overall search space, which might be purely discrete, purely
    continuous, or hybrid.

    NOTE:
        Created objects related to the computational representations of parameters
        (e.g., parameter bounds, computational dataframes, etc.) may use a different
        parameter order than what is specified through the constructor: While the
        passed parameter list can contain parameters in arbitrary order, the
        aforementioned objects (by convention) list discrete parameters first, followed
        by continuous ones.
    """

    def __init__(
        self,
        parameters: List[Parameter],
        constraints: Optional[List[Constraint]] = None,
        empty_encoding: bool = False,
        init_dataframes: bool = True,
    ):
        """
        Parameters
        ----------
        parameters : List[Parameter]
            The parameters spanning the search space.
        constraints : List[Constraint], optional
            An optional set of constraints restricting the valid parameter space.
        empty_encoding : bool, default: False
            If True, uses an "empty" encoding for all parameters. This is useful,
            for instance, in combination with random search strategies that
            do not read the actual parameter values, since it avoids the
            (potentially costly) transformation of the parameter values to their
            computational representation.
        init_dataframes : bool, default: True
            If True, the search space related dataframes (i.e. parameter representations
            and metadata) will be build from scratch using the input arguments. If
            False, they are not initialized, which can be useful when loading a search
            space object from disk.
        """
        # Store the input
        self.parameters = parameters
        self.constraints = constraints
        self.empty_encoding = empty_encoding

        self.discrete: SubspaceDiscrete = SubspaceDiscrete(
            parameters=[p for p in parameters if p.is_discrete],
            constraints=constraints,
            empty_encoding=empty_encoding,
            init_dataframes=init_dataframes,
        )
        self.continuous: SubspaceContinuous = SubspaceContinuous(
            parameters=[p for p in parameters if not p.is_discrete],
        )

    @property
    def contains_mordred(self) -> bool:
        """Indicates if any of the discrete parameters uses MORDRED encoding."""
        return any(p.encoding == "MORDRED" for p in self.discrete.parameters)

    @property
    def contains_rdkit(self) -> bool:
        """Indicates if any of the discrete parameters uses RDKIT encoding."""
        return any(p.encoding == "RDKIT" for p in self.discrete.parameters)

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """List of parameter bounds."""
        disc_bounds = self.discrete.bounds
        cont_bounds = self.continuous.bounds
        return [*disc_bounds, *cont_bounds]

    @property
    def tensor_bounds(self) -> torch.Tensor:
        """
        Returns bounds as tensor.
        """
        return torch.vstack(
            [self.discrete.tensor_bounds, self.continuous.tensor_bounds]
        )

    def state_dict(self) -> dict:
        """Creates a dictionary representing the object's internal state."""
        state_dict = dict(
            empty_encoding=self.empty_encoding,
            discrete=self.discrete.state_dict(),
        )
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Restores a given object state."""
        self.empty_encoding = state_dict["empty_encoding"]
        self.discrete.load_state_dict(state_dict["discrete"])

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transforms data (such as the measurements) from experimental to computational
        representation. Continuous parameters are not transformed but included.

        Parameters
        ----------
        data : pd.DataFrame
            The data to be transformed. Must contain all specified parameters, can
            contain more columns.

        Returns
        -------
        pd.DataFrame
            A dataframe with the parameters in computational representation.
        """
        # Transform subspaces separately
        df_discrete = self.discrete.transform(data)
        df_continuous = self.continuous.transform(data)

        # Combine Subspaces
        comp_rep = pd.concat([df_discrete, df_continuous], axis=1)

        return comp_rep


class SubspaceDiscrete:
    """
    Class for managing discrete search spaces.

    Builds the search space from parameter definitions and optional constraints, keeps
    track of search metadata, and provides access to candidate sets and different
    parameter views.
    """

    def __init__(
        self,
        parameters: List[DiscreteParameter],
        constraints: Optional[List[Constraint]] = None,
        empty_encoding: bool = False,
        init_dataframes: bool = True,
    ):
        """See `SearchSpace` class."""
        # Store the input
        self.parameters = parameters
        self.empty_encoding = empty_encoding
        if constraints is None:
            self.constraints = []
        else:
            # Reorder the constraints according to their execution order
            self.constraints = sorted(
                constraints, key=lambda x: _constraints_order.index(x.type)
            )

        self.exp_rep: Optional[pd.DataFrame] = None
        self.comp_rep: Optional[pd.DataFrame] = None
        self.metadata: Optional[pd.DataFrame] = None

        # Initialize discrete search space dataframes
        if init_dataframes:
            # Create a dataframe representing the experimental search space
            self.exp_rep = parameter_cartesian_prod_to_df(parameters)

            # Remove entries that violate parameter constraints:
            for constraint in (c for c in self.constraints if c.eval_during_creation):
                inds = constraint.get_invalid(self.exp_rep)
                self.exp_rep.drop(index=inds, inplace=True)
            self.exp_rep.reset_index(inplace=True, drop=True)

            # Create a dataframe containing the computational parameter representation
            # (ignoring all columns that do not carry any covariate information).
            self.comp_rep = self.transform(self.exp_rep)
            self.comp_rep = df_drop_single_value_columns(self.comp_rep)

            # Create a dataframe storing the experiment metadata
            self.metadata = pd.DataFrame(
                {
                    "was_recommended": False,
                    "was_measured": False,
                    "dont_recommend": False,
                },
                index=self.exp_rep.index,
            )

    @property
    def empty(self):
        """Whether this search space is empty."""
        return len(self.parameters) == 0

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """
        Returns list of parameter bounds.
        """
        # TODO: introduce bounds property for parameters
        # TODO: At the moment, this property exists mostly for consistency to have a
        #   counterpart to the version in SubspaceContinuous. Do we really need the
        #   bounds methods at all and does a list-based version even make sense for
        #   discrete parameters, which can have multidimensional computational reps?
        return [tuple(b) for b in self.tensor_bounds.tolist()]

    @property
    def tensor_bounds(self) -> torch.Tensor:
        """
        Returns bounds as tensor.
        """
        if not self.parameters:
            return torch.empty(0, 2)
        bounds = np.vstack(
            [
                np.c_[p.comp_df.min().values, p.comp_df.max().values]
                for p in self.parameters
            ]
        )
        return torch.from_numpy(bounds)

    def state_dict(self) -> dict:
        """Creates a dictionary representing the object's internal state."""
        state_dict = dict(
            empty_encoding=self.empty_encoding,
            exp_rep=self.exp_rep,
            comp_rep=self.comp_rep,
            metadata=self.metadata,
        )
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Restores a given object state."""
        self.empty_encoding = state_dict["empty_encoding"]
        self.exp_rep = state_dict["exp_rep"]
        self.comp_rep = state_dict["comp_rep"]
        self.metadata = state_dict["metadata"]

    def mark_as_measured(
        self,
        measurements: pd.DataFrame,
        numerical_measurements_must_be_within_tolerance: bool,
    ) -> None:
        """
        Marks the given elements of the search space as measured.

        Parameters
        ----------
        measurements : pd.DataFrame
            A dataframe containing parameter settings that should be marked as measured.
        numerical_measurements_must_be_within_tolerance : bool
            See `_match_measurement_with_searchspace_indices`.

        Returns
        -------
        Nothing.
        """
        inds_matched = self._match_measurement_with_searchspace_indices(
            measurements,
            numerical_measurements_must_be_within_tolerance,
        )
        self.metadata.loc[inds_matched, "was_measured"] = True

    def _match_measurement_with_searchspace_indices(
        self,
        df: pd.DataFrame,
        numerical_measurements_must_be_within_tolerance: bool,
    ) -> pd.Index:
        """
        Matches rows of a dataframe (e.g. measurements from an experiment)
        to the indices of the search space dataframe.

        This is useful for validity checks and to automatically match measurements to
        entries in the search space, e.g. to detect which ones have been measured.
        For categorical parameters, there needs to be an exact match with any of the
        allowed values. For numerical parameters, the user can decide via a flag
        whether values outside the tolerance should be accepted.

        Parameters
        ----------
        df : pd.DataFrame
            The data that should be checked for matching entries in the search space.
        numerical_measurements_must_be_within_tolerance : bool
            If True, numerical parameters are matched with the search space elements
            only if there is a match within the parameter tolerance. If False,
            the closest match is considered, irrespective of the distance.

        Returns
        -------
        pd.Index
            The index of the matching search space entries.
        """
        # IMPROVE: neater implementation (e.g. via fuzzy join)

        # Assert that all parameters appear in the given dataframe
        if not all(col in df.columns for col in self.exp_rep.columns):
            raise ValueError(
                "Values for all parameter must be specified in the given dataframe."
            )

        inds_matched = []

        # Iterate over all input rows
        for ind, row in df.iterrows():

            # Check if the row represents a valid input
            valid = True
            for param in self.parameters:
                if param.is_numeric:
                    if numerical_measurements_must_be_within_tolerance:
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

            # Differentiate category-like and discrete numerical parameters
            cat_cols = [p.name for p in self.parameters if not p.is_numeric]
            num_cols = [
                p.name for p in self.parameters if (p.is_numeric and p.is_discrete)
            ]

            # Discrete parameters must match exactly
            match = self.exp_rep[cat_cols].eq(row[cat_cols]).all(axis=1, skipna=False)

            # For numeric parameters, match the entry with the smallest deviation
            # TODO: allow alternative distance metrics
            for param in num_cols:
                abs_diff = (self.exp_rep[param] - row[param]).abs()
                match &= abs_diff == abs_diff.min()

            # We expect exactly one match. If that's not the case, print a warning.
            inds_found = self.exp_rep.index[match].to_list()
            if len(inds_found) == 0 and len(num_cols) > 0:
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

    def get_candidates(
        self,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns the set of candidate parameter settings that can be tested.

        Parameters
        ----------
        allow_repeated_recommendations : bool
            If True, parameter settings that have already been recommended in an
            earlier iteration are still considered as valid candidates. This is
            relevant, for instance, when an earlier recommended parameter setting has
            not been measured by the user (for any reason) after the corresponding
            recommendation was made.
        allow_recommending_already_measured : bool
            If True, parameters settings for which there are already target values
            available are still considered as valid candidates.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The candidate parameter settings both in experimental and computational
            representation.
        """
        # Filter the search space down to the candidates
        mask_todrop = self.metadata["dont_recommend"].copy()
        if not allow_repeated_recommendations:
            mask_todrop |= self.metadata["was_recommended"]
        if not allow_recommending_already_measured:
            mask_todrop |= self.metadata["was_measured"]

        return self.exp_rep.loc[~mask_todrop], self.comp_rep.loc[~mask_todrop]

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transforms discrete parameters from experimental to computational
        representation. Continuous parameters and additional columns are ignored.

        Parameters
        ----------
        data : pd.DataFrame
            The data to be transformed. Must contain all specified parameters, can
            contain more columns.

        Returns
        -------
        pd.DataFrame
            A dataframe with the parameters in computational representation.
        """
        # If the transformed values are not required, return an empty dataframe
        if self.empty_encoding or len(data) < 1:
            comp_rep = pd.DataFrame(index=data.index)
            return comp_rep

        # Transform the parameters
        dfs = []
        for param in self.parameters:
            comp_df = param.transform_rep_exp2comp(data[param.name])
            dfs.append(comp_df)
        comp_rep = pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

        # IMPROVE: The following is a simple mechanism to implement statefulness of
        #   the transformation. However, the state is effectively implemented through
        #   the presence of the `comp_rep` member and hence outside of this function.
        #   Also, the effective column subset is determined outside of this function.
        #   A slightly better alternative would be to use transformation object that
        #   internally keeps track of the state.
        try:
            # If this does not raise an error, the search space constructor has
            # already been called and the columns to be kept are known. Hence,
            # every subsequent transformation should adhere to the same column subset.
            comp_rep = comp_rep[self.comp_rep.columns]
        except AttributeError:
            # Otherwise, the transformation is being called by the search space
            # constructor so that no column filtering is needed.
            pass

        return comp_rep


class SubspaceContinuous:
    """
    Class for managing continuous search spaces.
    """

    def __init__(
        self,
        parameters: List[NumericContinuous],
    ):
        """See `SearchSpace` class."""
        self.parameters: List[NumericContinuous] = parameters

    @property
    def empty(self):
        """Whether this search space is empty."""
        return len(self.parameters) == 0

    @property
    def param_names(self) -> List[str]:
        """
        Returns list of parameter names.
        """
        return [p.name for p in self.parameters]

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """
        Returns list of parameter bounds.
        """
        return [p.bounds for p in self.parameters]

    @property
    def bounds_forced_finite(self) -> List[Tuple[float, float]]:
        """
        Returns list of parameter bounds. Infinite will be replaced by 1000.
        """
        finite_bounds = list(
            map(
                lambda x: (
                    x[0] if np.isfinite(x[0]) else -INF_BOUNDS_REPLACEMENT,
                    x[1] if np.isfinite(x[1]) else INF_BOUNDS_REPLACEMENT,
                ),
                self.bounds,
            )
        )

        return finite_bounds

    @property
    def is_fully_bounded(self):
        """
        Whether the search space has infinite bound or is entirely finitely bounded.

        Returns
        -------
        bool
            True if search space has no infinite bounds.
        """
        return self.bounds == self.bounds_forced_finite

    @property
    def tensor_bounds(self) -> torch.Tensor:
        """
        Returns bounds as tensor.
        """
        if not self.parameters:
            return torch.empty(0, 2)
        return torch.tensor(self.bounds)

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        See SubspaceDiscrete.transform
        """
        # Transform continuous parameters
        comp_rep = data[[p.name for p in self.parameters]]

        return comp_rep

    # TODO rework the helper functions below, remove finite bound enforcement and
    #  replace by distributional sampling
    def samples_random(self, n_points: int = 1) -> pd.DataFrame:
        """
        Get random point samples from the continuous space. Infinite bounds are
        replaced by half of the maximum floating point number.

        Parameters
        ----------
        n_points : int
            Number of points that should be sampled.

        Returns
        -------
        pandas data frame
            A data frame containing the points as rows with columns corresponding to the
             parameter names.
        """
        if len(self.parameters) < 1:
            return pd.DataFrame()

        vals = np.stack(
            [
                (b[1] - b[0]) * np.random.random(size=n_points) + b[0]
                for b in self.bounds_forced_finite
            ]
        ).T
        ret = pd.DataFrame(vals, columns=self.param_names)

        return ret

    def samples_full_factorial(self, n_points: int = 1) -> pd.DataFrame:
        """
        Get random point samples from the full factorial of the continuous space.

        Parameters
        ----------
        n_points : int
            Number of points that should be sampled.

        Returns
        -------
        pandas data frame
            A data frame containing the points as rows with columns corresponding to the
             parameter names.
        """
        full_factorial = self.full_factorial

        if len(full_factorial) < n_points:
            raise ValueError(
                f"You are trying to sample {n_points} points from the full factorial of"
                f" the continuous space bounds, but it has only {len(full_factorial)} "
                f"points."
            )

        return full_factorial.sample(n=n_points).reset_index(drop=True)

    @property
    def full_factorial(self) -> pd.DataFrame:
        """
        Get the full factorial of the continuous space.

        Returns
        -------
        pandas data frame
            A data frame containing the full factorial
        """
        if not self.is_fully_bounded:
            log.warning(
                "You are trying to access the full factorial of a continuous sace that "
                "has infinite bounds in at least one parameter. Internally, infinite "
                "bounds have been replaced by -/+ %f",
                INF_BOUNDS_REPLACEMENT,
            )

        index = pd.MultiIndex.from_product(
            self.bounds_forced_finite, names=self.param_names
        )

        return pd.DataFrame(index=index).reset_index()
