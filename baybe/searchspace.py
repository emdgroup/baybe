# pylint: disable=missing-class-docstring, missing-function-docstring
# TODO: add docstrings

"""
Functionality for managing search spaces.
"""
# TODO: ForwardRefs via __future__ annotations are currently disabled due to this issue:
#  https://github.com/python-attrs/cattrs/issues/354

from enum import Enum
from typing import cast, Dict, List, Optional, Tuple

import cattrs
import numpy as np
import pandas as pd
import torch
from attrs import define, field

from baybe.constraints import _validate_constraints, Constraint, CONSTRAINTS_ORDER
from baybe.parameters import (
    _validate_parameter_names,
    _validate_parameters,
    Categorical,
    DiscreteParameter,
    NumericContinuous,
    NumericDiscrete,
    Parameter,
    parameter_cartesian_prod_to_df,
)
from baybe.telemetry import TELEM_LABELS, telemetry_record_value
from baybe.utils.boolean import eq_dataframe
from baybe.utils.dataframe import df_drop_single_value_columns, fuzzy_row_match
from baybe.utils.serialization import SerialMixin


class SearchSpaceType(Enum):
    DISCRETE = "DISCRETE"
    CONTINUOUS = "CONTINUOUS"
    EITHER = "EITHER"
    HYBRID = "HYBRID"


@define
class SubspaceDiscrete:
    """
    Class for managing discrete search spaces.

    Builds the search space from parameter definitions and optional constraints, keeps
    track of search metadata, and provides access to candidate sets and different
    parameter views.
    """

    parameters: List[DiscreteParameter] = field(
        validator=lambda _1, _2, x: _validate_parameter_names(x)
    )
    exp_rep: pd.DataFrame = field(eq=eq_dataframe())
    comp_rep: pd.DataFrame = field(init=False, eq=eq_dataframe())
    metadata: pd.DataFrame = field(eq=eq_dataframe())
    empty_encoding: bool = field(default=False)
    constraints: List[Constraint] = field(factory=list)

    @metadata.default
    def default_metadata(self) -> pd.DataFrame:
        columns = ["was_recommended", "was_measured", "dont_recommend"]

        # If the discrete search space is empty, explicitly return an empty dataframe
        # instead of simply using a zero-length index. Otherwise, the boolean dtype
        # would be lost during a serialization roundtrip as there would be no
        # data available that allows to determine the type, causing subsequent
        # equality checks to fail.
        if self.is_empty:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(False, columns=columns, index=self.exp_rep.index)

    def __attrs_post_init__(self):
        # Create a dataframe containing the computational parameter representation
        # (ignoring all columns that do not carry any covariate information).
        # TODO[12758]: Should we always drop single value columns without informing the
        #  user? Can have undesired/unexpected side-effects (see ***REMOVED*** project).
        comp_rep = self.transform(self.exp_rep)
        comp_rep = df_drop_single_value_columns(comp_rep)
        self.comp_rep = comp_rep

    @classmethod
    def empty(cls) -> "SubspaceDiscrete":
        """Creates an empty discrete subspace."""
        return SubspaceDiscrete(
            parameters=[], exp_rep=pd.DataFrame(), metadata=pd.DataFrame()
        )

    @classmethod
    def from_product(
        cls,
        parameters: List[DiscreteParameter],
        constraints: Optional[List[Constraint]] = None,
        empty_encoding: bool = False,
    ) -> "SubspaceDiscrete":
        """See `SearchSpace` class."""
        # Store the input
        if constraints is None:
            constraints = []
        else:
            # Reorder the constraints according to their execution order
            constraints = sorted(
                constraints, key=lambda x: CONSTRAINTS_ORDER.index(x.__class__)
            )

        # Create a dataframe representing the experimental search space
        exp_rep = parameter_cartesian_prod_to_df(parameters)

        # Remove entries that violate parameter constraints:
        for constraint in (c for c in constraints if c.eval_during_creation):
            inds = constraint.get_invalid(exp_rep)
            exp_rep.drop(index=inds, inplace=True)
        exp_rep.reset_index(inplace=True, drop=True)

        return SubspaceDiscrete(
            parameters=parameters,
            constraints=constraints,
            exp_rep=exp_rep,
            empty_encoding=empty_encoding,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        parameters: Optional[List[Parameter]] = None,
        empty_encoding: bool = False,
    ) -> "SubspaceDiscrete":
        """
        Creates a discrete subspace with a specified set of configurations.

        Parameters
        ----------
        df : pd.DataFrame
            The experimental representation of the search space to be created.
        parameters : pd.DataFrame
            Optional parameters corresponding to the columns in the given dataframe.
            If a match between column name and parameter name is found, the
            corresponding parameter is used. If a column has no match in the parameter
            list, a `NumericDiscrete` parameter is created if possible, or a
            `Categorical` is used as fallback.
        empty_encoding : bool
            See `SearchSpace` class.

        Returns
        -------
        SubspaceDiscrete
            The created discrete subspace.
        """
        # Turn the specified parameters into a dict and check for duplicate names
        specified_params: Dict[str, Parameter] = {}
        if parameters is not None:
            for param in parameters:
                if param.name in specified_params:
                    raise ValueError(
                        f"You provided several parameters with the name {param.name}."
                    )
                specified_params[param.name] = param

        # Try to find a parameter match for each dataframe column
        parameters = []
        for name, series in df.iteritems():

            # If a match is found, assert that the values are in range
            if match := specified_params.pop(name, None):
                assert series.apply(match.is_in_range).all()
                parameters.append(match)

            # Otherwise, try to create a numerical parameter or use categorical fallback
            else:
                values = series.drop_duplicates().values.tolist()
                try:
                    param = NumericDiscrete(name=name, values=values)
                except TypeError:
                    param = Categorical(name=name, values=values)
                parameters.append(param)

        # By now, all parameters must have been used
        if specified_params:
            raise ValueError(
                f"For the following parameters you specified, no match could be found "
                f"in the given dataframe: {specified_params.values()}."
            )

        return SubspaceDiscrete(
            parameters=parameters, exp_rep=df, empty_encoding=empty_encoding
        )

    @property
    def is_empty(self):
        """Whether this search space is empty."""
        return len(self.parameters) == 0

    @property
    def param_bounds_comp(self) -> torch.Tensor:
        """
        Returns bounds as tensor. Takes bounds from the parameter definitions, but
        discards bounds belonging to columns that were filtered out during search space
        creation.
        """
        if not self.parameters:
            return torch.empty(2, 0)
        bounds = np.hstack(
            [
                np.vstack([p.comp_df[col].min(), p.comp_df[col].max()])
                for p in self.parameters
                for col in p.comp_df
                if col in self.comp_rep.columns
            ]
        )
        return torch.from_numpy(bounds)

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
            See utility `fuzzy_row_match`.

        Returns
        -------
        Nothing.
        """
        inds_matched = fuzzy_row_match(
            self.exp_rep,
            measurements,
            self.parameters,
            numerical_measurements_must_be_within_tolerance,
        )
        self.metadata.loc[inds_matched, "was_measured"] = True

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

        # If the computational representation has already been built (with potentially
        # removing some columns, e.g. due to decorrelation or dropping constant ones),
        # any subsequent transformation should yield the same columns.
        try:
            comp_rep = comp_rep[self.comp_rep.columns]
        except AttributeError:
            pass

        return comp_rep


@define
class SubspaceContinuous:
    """
    Class for managing continuous search spaces.
    """

    parameters: List[NumericContinuous] = field(
        validator=lambda _1, _2, x: _validate_parameter_names(x)
    )

    @classmethod
    def empty(cls) -> "SubspaceContinuous":
        """Creates an empty continuous subspace."""
        return SubspaceContinuous([])

    @classmethod
    def from_bounds(cls, bounds: pd.DataFrame) -> "SubspaceContinuous":
        """Creates a hyperrectangle-shaped continuous search space with given bounds."""

        # Assert that the input represents valid bounds
        assert bounds.shape[0] == 2
        assert (np.diff(bounds.values, axis=0) >= 0).all()
        assert bounds.apply(pd.api.types.is_numeric_dtype).all()

        # Create the corresponding parameters and from them the search space
        parameters = [
            NumericContinuous(name, bound) for (name, bound) in bounds.iteritems()
        ]
        return SubspaceContinuous(parameters)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "SubspaceContinuous":
        """
        Creates the smallest axis-aligned hyperrectangle-shaped continuous subspace
        that contains the points specified in the given dataframe.
        """
        # TODO: Add option for convex hull once constraints are in place
        bounds = pd.concat([df.min(), df.max()], axis=1).T
        return cls.from_bounds(bounds)

    @property
    def is_empty(self):
        """Whether this search space is empty."""
        return len(self.parameters) == 0

    @property
    def param_names(self) -> List[str]:
        """
        Returns list of parameter names.
        """
        return [p.name for p in self.parameters]

    @property
    def param_bounds_comp(self) -> torch.Tensor:
        """
        Returns bounds as tensor.
        """
        if not self.parameters:
            return torch.empty(2, 0)
        return torch.stack([p.bounds.to_tensor() for p in self.parameters]).T

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
        if not self.parameters:
            return pd.DataFrame()
        points = torch.distributions.uniform.Uniform(*self.param_bounds_comp).sample(
            torch.Size((n_points,))
        )
        return pd.DataFrame(points, columns=self.param_names)

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
        index = pd.MultiIndex.from_product(
            self.param_bounds_comp.T.tolist(), names=self.param_names
        )

        return pd.DataFrame(index=index).reset_index()


@define
class SearchSpace(SerialMixin):
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

    discrete: SubspaceDiscrete = field(factory=SubspaceDiscrete.empty)
    continuous: SubspaceContinuous = field(factory=SubspaceContinuous.empty)

    def __attrs_post_init__(self):
        _validate_parameters(self.parameters)
        _validate_constraints(self.discrete.constraints)

        # Telemetry
        telemetry_record_value(TELEM_LABELS["COUNT_SEARCHSPACE_CREATION"], 1)
        telemetry_record_value(TELEM_LABELS["NUM_PARAMETERS"], len(self.parameters))
        telemetry_record_value(
            TELEM_LABELS["NUM_CONSTRAINTS"],
            len(self.constraints) if self.constraints else 0,
        )

    @classmethod
    def from_product(
        cls,
        parameters: List[Parameter],
        constraints: Optional[List[Constraint]] = None,
        empty_encoding: bool = False,
    ) -> "SearchSpace":
        """
        Creates a "product" search space (with optional subsequent constraints applied).

        That is, the discrete subspace becomes the (filtered) cartesian product
        containing all discrete parameter combinations while, analogously, the
        continuous subspace represents the (filtered) cartesian product of all
        continuous parameters. (TODO: continuous constraints are yet to be enabled.)

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
        """
        # IMPROVE: The arguments get pre-validated here to avoid the potentially costly
        #   creation of the subspaces. Perhaps there is an elegant way to bypass the
        #   default validation in the initializer (which is required for other
        #   ways of object creation) in this particular case.
        _validate_parameters(parameters)
        if constraints:
            _validate_constraints(constraints)

        discrete: SubspaceDiscrete = SubspaceDiscrete.from_product(
            parameters=[
                cast(DiscreteParameter, p) for p in parameters if p.is_discrete
            ],
            constraints=constraints,
            empty_encoding=empty_encoding,
        )
        continuous: SubspaceContinuous = SubspaceContinuous(
            parameters=[
                cast(NumericContinuous, p) for p in parameters if not p.is_discrete
            ],
        )

        return SearchSpace(discrete=discrete, continuous=continuous)

    @property
    def parameters(self) -> List[Parameter]:
        return self.discrete.parameters + self.continuous.parameters

    @property
    def constraints(self) -> List[Constraint]:
        return self.discrete.constraints

    @property
    def type(self) -> SearchSpaceType:
        if self.discrete.is_empty and not self.continuous.is_empty:
            return SearchSpaceType.CONTINUOUS
        if not self.discrete.is_empty and self.continuous.is_empty:
            return SearchSpaceType.DISCRETE
        if not self.discrete.is_empty and not self.continuous.is_empty:
            return SearchSpaceType.HYBRID
        raise RuntimeError("This line should be impossible to reach.")

    @property
    def contains_mordred(self) -> bool:
        """Indicates if any of the discrete parameters uses MORDRED encoding."""
        return any(p.encoding == "MORDRED" for p in self.discrete.parameters)

    @property
    def contains_rdkit(self) -> bool:
        """Indicates if any of the discrete parameters uses RDKIT encoding."""
        return any(p.encoding == "RDKIT" for p in self.discrete.parameters)

    @property
    def param_bounds_comp(self) -> torch.Tensor:
        """
        Returns bounds as tensor.
        """
        return torch.hstack(
            [self.discrete.param_bounds_comp, self.continuous.param_bounds_comp]
        )

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


# TODO: The following structuring hook is a workaround for field with init=False.
#   https://github.com/python-attrs/cattrs/issues/40


def structure_hook(dict_, type_):
    dict_.pop("comp_rep")
    return cattrs.structure_attrs_fromdict(dict_, type_)


cattrs.register_structure_hook(SubspaceDiscrete, structure_hook)
