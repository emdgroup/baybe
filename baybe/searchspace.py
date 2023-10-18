"""Functionality for managing search spaces."""

from __future__ import annotations

from enum import Enum
from typing import Any, cast, Dict, List, Optional, Tuple

import cattrs
import numpy as np
import pandas as pd
import torch
from attrs import define, field
from botorch.utils.sampling import get_polytope_samples
from cattrs.errors import IterableValidationError

from baybe.constraints import (
    _validate_constraints,
    Constraint,
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
    DISCRETE_CONSTRAINTS_FILTERING_ORDER,
    DiscreteConstraint,
)
from baybe.parameters import (
    _validate_parameter_names,
    _validate_parameters,
    CategoricalParameter,
    DiscreteParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    Parameter,
    parameter_cartesian_prod_to_df,
    TaskParameter,
)
from baybe.telemetry import TELEM_LABELS, telemetry_record_value
from baybe.utils import (
    df_drop_single_value_columns,
    DTypeFloatTorch,
    eq_dataframe,
    fuzzy_row_match,
    SerialMixin,
)

_METADATA_COLUMNS = ["was_recommended", "was_measured", "dont_recommend"]


class SearchSpaceType(Enum):
    """Enum class for different types of search spaces and respective compatibility."""

    DISCRETE = "DISCRETE"
    """Flag for discrete search spaces resp. compatibility with discrete search
    spaces."""
    CONTINUOUS = "CONTINUOUS"
    """Flag for continuous search spaces resp. compatibility with continuous
    search spaces."""
    EITHER = "EITHER"
    """Flag compatibility with either discrete or continuous, but not hybrid
    search spaces."""
    HYBRID = "HYBRID"
    """Flag for hybrid search spaces resp. compatibility with hybrid search spaces."""


@define
class SubspaceDiscrete:
    """Class for managing discrete subspaces.

    Builds the subspace from parameter definitions and optional constraints, keeps
    track of search metadata, and provides access to candidate sets and different
    parameter views.

    Args:
        parameters: The list of parameters of the subspace.
        exp_rep: The experimental representation of the subspace.
        metadata: The metadata.
        empty_encoding: Flag encoding whether an empty encoding is used.
        constraints: A list of constraints for restricting the space.
        comp_rep: The computational representation of the space.
            Technically not required but added as an optional initializer argument to
            allow ingestion from e.g. serialized objects and thereby speed up
            construction. If not provided, the default hook will derive it from
            ```exp_rep```.
    """

    parameters: List[DiscreteParameter] = field(
        validator=lambda _1, _2, x: _validate_parameter_names(x)
    )
    exp_rep: pd.DataFrame = field(eq=eq_dataframe)
    metadata: pd.DataFrame = field(eq=eq_dataframe)
    empty_encoding: bool = field(default=False)
    constraints: List[DiscreteConstraint] = field(factory=list)
    comp_rep: pd.DataFrame = field(eq=eq_dataframe)

    @exp_rep.validator
    def _validate_exp_rep(  # noqa: DOC101, DOC103
        self, _: Any, exp_rep: pd.DataFrame
    ) -> None:
        """Validate the experimental representation.

        Raises:
            ValueError: If the index of the provided dataframe contains duplicates.
        """
        if exp_rep.index.has_duplicates:
            raise ValueError(
                "The index of this search space contains duplicates. "
                "This is not allowed, as it can lead to hard-to-detect bugs."
            )

    @metadata.default
    def _default_metadata(self) -> pd.DataFrame:
        """Create the default metadata."""
        # If the discrete search space is empty, explicitly return an empty dataframe
        # instead of simply using a zero-length index. Otherwise, the boolean dtype
        # would be lost during a serialization roundtrip as there would be no
        # data available that allows to determine the type, causing subsequent
        # equality checks to fail.
        # TODO: verify if this is still required
        if self.is_empty:
            return pd.DataFrame(columns=_METADATA_COLUMNS)

        # TODO [16605]: Redesign metadata handling
        # Exclude inactive tasks from search
        df = pd.DataFrame(False, columns=_METADATA_COLUMNS, index=self.exp_rep.index)
        off_task_idxs = ~self._on_task_configurations()
        df.loc[off_task_idxs.values, "dont_recommend"] = True
        return df

    @metadata.validator
    def _validate_metadata(  # noqa: DOC101, DOC103
        self, _: Any, metadata: pd.DataFrame
    ) -> None:
        """Validate the metadata.

        Raises:
            ValueError: If the provided metadata allows testing parameter configurations
                for inactive tasks.
        """
        off_task_idxs = ~self._on_task_configurations()
        if not metadata.loc[off_task_idxs.values, "dont_recommend"].all():
            raise ValueError(
                "Inconsistent instructions given: The provided metadata allows "
                "testing parameter configurations for inactive tasks."
            )

    @comp_rep.default
    def _default_comp_rep(self) -> pd.DataFrame:
        """Create the default computational representation."""
        # Create a dataframe containing the computational parameter representation
        comp_rep = self.transform(self.exp_rep)

        # Ignore all columns that do not carry any covariate information
        # TODO[12758]: Should we always drop single value columns without informing the
        #  user? Can have undesired/unexpected side-effects (see ***REMOVED*** project).
        comp_rep = df_drop_single_value_columns(comp_rep)

        return comp_rep

    def __attrs_post_init__(self) -> None:
        # TODO [16605]: Redesign metadata handling
        off_task_idxs = ~self._on_task_configurations()
        self.metadata.loc[off_task_idxs.values, "dont_recommend"] = True

    def _on_task_configurations(self) -> pd.Series:
        """Retrieves the parameter configurations for the active tasks."""
        # TODO [16932]: This only works for a single parameter
        try:
            task_param = next(
                p for p in self.parameters if isinstance(p, TaskParameter)
            )
        except StopIteration:
            return pd.Series(True, index=self.exp_rep.index)
        return self.exp_rep[task_param.name].isin(task_param.active_values)

    @classmethod
    def empty(cls) -> SubspaceDiscrete:
        """Creates an empty discrete subspace."""
        return SubspaceDiscrete(
            parameters=[],
            exp_rep=pd.DataFrame(),
            metadata=pd.DataFrame(columns=_METADATA_COLUMNS),
        )

    @classmethod
    def from_product(
        cls,
        parameters: List[DiscreteParameter],
        constraints: Optional[List[DiscreteConstraint]] = None,
        empty_encoding: bool = False,
    ) -> SubspaceDiscrete:
        """See :class:`baybe.searchspace.SearchSpace`."""
        # Store the input
        if constraints is None:
            constraints = []
        else:
            # Reorder the constraints according to their execution order
            constraints = sorted(
                constraints,
                key=lambda x: DISCRETE_CONSTRAINTS_FILTERING_ORDER.index(x.__class__),
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
    ) -> SubspaceDiscrete:
        """Create a discrete subspace with a specified set of configurations.

        Args:
            df: The experimental representation of the search space to be created.
            parameters: Optional parameters corresponding to the columns in the given
                dataframe. If a match between column name and parameter name is found,
                the corresponding parameter is used. If a column has no match in the
                parameter list, a
                :class:`baybe.parameters.NumericalDiscreteParameter` is created if
                possible, or a :class:`baybe.parameters.CategoricalParameter` is used
                as fallback.
            empty_encoding: See :class:`baybe.searchspace.SearchSpace`.

        Returns:
            The created discrete subspace.

        Raises:
            ValueError: If several parameters with identical names are provided.
            ValueError: If a parameter was specified for which no match was found.
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
        for name, series in df.items():

            # If a match is found, assert that the values are in range
            if match := specified_params.pop(name, None):
                assert series.apply(match.is_in_range).all()
                parameters.append(match)

            # Otherwise, try to create a numerical parameter or use categorical fallback
            else:
                values = series.drop_duplicates().values.tolist()
                try:
                    param = NumericalDiscreteParameter(name=name, values=values)
                except IterableValidationError:
                    param = CategoricalParameter(name=name, values=values)
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
    def is_empty(self) -> bool:
        """Return whether this subspace is empty."""
        return len(self.parameters) == 0

    @property
    def param_bounds_comp(self) -> torch.Tensor:
        """Return bounds as tensor.

        Take bounds from the parameter definitions, but discards bounds belonging to
        columns that were filtered out during the creation of the space.
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
        """Mark the given elements of the space as measured.

        Args:
            measurements: A dataframe containing parameter settings that should be
                marked as measured.
            numerical_measurements_must_be_within_tolerance: See
                :func:`baybe.utils.dataframe.fuzzy_row_match`.
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
        """Return the set of candidate parameter settings that can be tested.

        Args:
            allow_repeated_recommendations: If ```True```, parameter settings that have
                already been recommended in an earlier iteration are still considered
                valid candidates. This is relevant, for instance, when an earlier
                recommended parameter setting has not been measured by the user (for any
                reason) after the corresponding recommendation was made.
            allow_recommending_already_measured: If ```True```, parameters settings for
                which there are already target values available are still considered as
                valid candidates.

        Returns:
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
        """Transform parameters from experimental to computational representation.

        Continuous parameters and additional columns are ignored.

        Args:
            data: The data to be transformed. Must contain all specified parameters, can
                contain more columns.

        Returns:
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
    """Class for managing continuous subspaces.

    Builds the subspace from parameter definitions, keeps
    track of search metadata, and provides access to candidate sets and different
    parameter views.

    Args:
        parameters: The list of parameters of the subspace.
        constraints_lin_eq: List of linear equality constraints.
        constraints_lin_ineq: List of linear inequality constraints.
    """

    parameters: List[NumericalContinuousParameter] = field(
        validator=lambda _1, _2, x: _validate_parameter_names(x)
    )
    constraints_lin_eq: List[ContinuousLinearEqualityConstraint] = field(factory=list)
    constraints_lin_ineq: List[ContinuousLinearInequalityConstraint] = field(
        factory=list
    )

    @classmethod
    def empty(cls) -> SubspaceContinuous:
        """Create an empty continuous subspace."""
        return SubspaceContinuous([])

    @classmethod
    def from_bounds(cls, bounds: pd.DataFrame) -> SubspaceContinuous:
        """Create a hyperrectangle-shaped continuous subspace with given bounds.

        Args:
            bounds: The bounds of the parameters.

        Returns:
            The constructed subspace.
        """
        # Assert that the input represents valid bounds
        assert bounds.shape[0] == 2
        assert (np.diff(bounds.values, axis=0) >= 0).all()
        assert bounds.apply(pd.api.types.is_numeric_dtype).all()

        # Create the corresponding parameters and from them the search space
        parameters = [
            NumericalContinuousParameter(name, bound)
            for (name, bound) in bounds.items()
        ]
        return SubspaceContinuous(parameters)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> SubspaceContinuous:
        """Create a hyperractangle-shaped continuous subspace from a dataframe.

        More precisely, create the smallest axis-aligned hyperrectangle-shaped
        continuous subspace that contains the points specified in the given dataframe.

        Args:
            df: The dataframe specifying the points of the subspace.

        Returns:
            Ths constructed subspace.
        """
        # TODO: Add option for convex hull once constraints are in place
        bounds = pd.concat([df.min(), df.max()], axis=1).T
        return cls.from_bounds(bounds)

    @property
    def is_empty(self) -> bool:
        """Return whether this subspace is empty."""
        return len(self.parameters) == 0

    @property
    def param_names(self) -> List[str]:
        """Return list of parameter names."""
        return [p.name for p in self.parameters]

    @property
    def param_bounds_comp(self) -> torch.Tensor:
        """Return bounds as tensor."""
        if not self.parameters:
            return torch.empty(2, 0, dtype=DTypeFloatTorch)
        return torch.stack([p.bounds.to_tensor() for p in self.parameters]).T

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """See :func:`baybe.searchspace.SubspaceDiscrete.transform`.

        Args:
            data: The data that should be transformed.

        Returns:
            The transformed data.
        """
        # Transform continuous parameters
        comp_rep = data[[p.name for p in self.parameters]]

        return comp_rep

    def samples_random(self, n_points: int = 1) -> pd.DataFrame:
        """Get random point samples from the continuous space.

        Args:
            n_points: Number of points that should be sampled.

        Returns:
            A data frame containing the points as rows with columns corresponding to the
            parameter names.
        """
        if not self.parameters:
            return pd.DataFrame()

        points = get_polytope_samples(
            n=n_points,
            bounds=self.param_bounds_comp,
            equality_constraints=[
                c.to_botorch(self.parameters) for c in self.constraints_lin_eq
            ],
            inequality_constraints=[
                c.to_botorch(self.parameters) for c in self.constraints_lin_ineq
            ],
        )

        return pd.DataFrame(points, columns=self.param_names)

    def samples_full_factorial(self, n_points: int = 1) -> pd.DataFrame:
        """Get random point samples from the full factorial of the continuous space.

        Args:
            n_points: Number of points that should be sampled.

        Returns:
            A data frame containing the points as rows with columns corresponding to the
            parameter names.

        Raises:
            ValueError: If there are not enough points to sample from.
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
        """Get the full factorial of the continuous space."""
        index = pd.MultiIndex.from_product(
            self.param_bounds_comp.T.tolist(), names=self.param_names
        )

        return pd.DataFrame(index=index).reset_index()


@define
class SearchSpace(SerialMixin):
    """Class for managing the overall search space.

    The search space might be purely discrete, purely continuous, or hybrid.
    Note that created objects related to the computational representations of parameters
    (e.g., parameter bounds, computational dataframes, etc.) may use a different
    parameter order than what is specified through the constructor: While the
    passed parameter list can contain parameters in arbitrary order, the
    aforementioned objects (by convention) list discrete parameters first, followed
    by continuous ones.

    Args:
        discrete: The (potentially empty) discrete subspace of the overall search space.
        continuous: The (potentially empty) continuous subspace of the overall
            search space.
    """

    discrete: SubspaceDiscrete = field(factory=SubspaceDiscrete.empty)
    continuous: SubspaceContinuous = field(factory=SubspaceContinuous.empty)

    def __attrs_post_init__(self):
        """Perform validation and record telemetry values."""
        _validate_parameters(self.parameters)
        _validate_constraints(self.constraints, self.parameters)

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
    ) -> SearchSpace:
        """Create a search space from a cartesian product.

        In the search space, optional subsequent constraints are applied.
        That is, the discrete subspace becomes the (filtered) cartesian product
        containing all discrete parameter combinations while, analogously, the
        continuous subspace represents the (filtered) cartesian product of all
        continuous parameters.

        Args:
            parameters: The parameters spanning the search space.
            constraints: An optional set of constraints restricting the valid parameter
                space.
            empty_encoding: If ```True```, uses an "empty" encoding for all parameters.
                This is useful, for instance, in combination with random search
                strategies that do not read the actual parameter values, since it avoids
                the (potentially costly) transformation of the parameter values to their
                computational representation.

        Returns:
            The constructed search space.
        """
        # IMPROVE: The arguments get pre-validated here to avoid the potentially costly
        #   creation of the subspaces. Perhaps there is an elegant way to bypass the
        #   default validation in the initializer (which is required for other
        #   ways of object creation) in this particular case.
        _validate_parameters(parameters)
        if constraints:
            _validate_constraints(constraints, parameters)
        else:
            constraints = []

        discrete: SubspaceDiscrete = SubspaceDiscrete.from_product(
            parameters=[
                cast(DiscreteParameter, p) for p in parameters if p.is_discrete
            ],
            constraints=[
                cast(DiscreteConstraint, c) for c in constraints if c.is_discrete
            ],
            empty_encoding=empty_encoding,
        )
        continuous: SubspaceContinuous = SubspaceContinuous(
            parameters=[
                cast(NumericalContinuousParameter, p)
                for p in parameters
                if not p.is_discrete
            ],
            constraints_lin_eq=[
                cast(ContinuousLinearEqualityConstraint, c)
                for c in constraints
                if isinstance(c, ContinuousLinearEqualityConstraint)
            ],
            constraints_lin_ineq=[
                cast(ContinuousLinearInequalityConstraint, c)
                for c in constraints
                if isinstance(c, ContinuousLinearInequalityConstraint)
            ],
        )

        return SearchSpace(discrete=discrete, continuous=continuous)

    @property
    def parameters(self) -> List[Parameter]:
        """Return the list of parameters of the search space."""
        return self.discrete.parameters + self.continuous.parameters

    @property
    def constraints(self) -> List[Constraint]:
        """Return the constraints of the search space."""
        return (
            self.discrete.constraints
            + self.continuous.constraints_lin_eq
            + self.continuous.constraints_lin_ineq
        )

    @property
    def type(self) -> SearchSpaceType:
        """Return the type of the search space."""
        if self.discrete.is_empty and not self.continuous.is_empty:
            return SearchSpaceType.CONTINUOUS
        if not self.discrete.is_empty and self.continuous.is_empty:
            return SearchSpaceType.DISCRETE
        if not self.discrete.is_empty and not self.continuous.is_empty:
            return SearchSpaceType.HYBRID
        raise RuntimeError("This line should be impossible to reach.")

    @property
    def contains_mordred(self) -> bool:
        """Indicates if any of the discrete parameters uses ```MORDRED``` encoding."""
        return any(p.encoding == "MORDRED" for p in self.discrete.parameters)

    @property
    def contains_rdkit(self) -> bool:
        """Indicates if any of the discrete parameters uses ```RDKIT``` encoding."""
        return any(p.encoding == "RDKIT" for p in self.discrete.parameters)

    @property
    def param_bounds_comp(self) -> torch.Tensor:
        """Return bounds as tensor."""
        return torch.hstack(
            [self.discrete.param_bounds_comp, self.continuous.param_bounds_comp]
        )

    @property
    def task_idx(self) -> Optional[int]:
        """The column index of the task parameter in computational representation."""
        try:
            # TODO [16932]: Redesign metadata handling
            task_param = next(
                p for p in self.parameters if isinstance(p, TaskParameter)
            )
        except StopIteration:
            return None
        # TODO[11611]: The current approach has two limitations:
        #   1.  It matches by column name and thus assumes that the parameter name
        #       is used as the column name.
        #   2.  It relies on the current implementation detail that discrete parameters
        #       appear first in the computational dataframe.
        #   --> Fix this when refactoring the data
        return self.discrete.comp_rep.columns.get_loc(task_param.name)

    @property
    def n_tasks(self) -> int:
        """The number of tasks encoded in the search space."""
        # TODO [16932]: This approach only works for a single task parameter. For
        #  multiple task parameters, we need to align what the output should even
        #  represent (e.g. number of combinatorial task combinations, number of
        #  tasks per task parameter, etc).
        try:
            task_param = next(
                p for p in self.parameters if isinstance(p, TaskParameter)
            )
            return len(task_param.values)

        # When there are no task parameters, we effectively have a single task
        except StopIteration:
            return 1

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform data from experimental to computational representation.

        This function can e.g. be used to transform data obtained from measurements.
        Continuous parameters are not transformed but included.

        Args:
            data: The data to be transformed. Must contain all specified parameters, can
                contain more columns.

        Returns:
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


def _structure_hook(dict_, type_):
    """Structuring hook for SubspaceDiscrete."""
    return cattrs.structure_attrs_fromdict(dict_, type_)


cattrs.register_structure_hook(SubspaceDiscrete, _structure_hook)
