"""Continuous subspaces."""

from __future__ import annotations

import random
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Any, Optional, cast
from typing_extensions import TypeAlias

import numpy as np
import pandas as pd
from attr import define, field

from baybe.constraints import (
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
    ContinuousCardinalityConstraint,
)
from baybe.parameters import NumericalContinuousParameter
from baybe.parameters.base import ContinuousParameter
from baybe.parameters.utils import get_parameters_from_dataframe
from baybe.searchspace.validation import validate_parameter_names
from baybe.constraints.validation import validate_continuous_cardinality_constraints
from baybe.serialization import SerialMixin, converter, select_constructor_hook
from baybe.utils.basic import to_tuple
from baybe.utils.dataframe import pretty_print_df
from baybe.utils.numerical import DTypeFloatNumpy

if TYPE_CHECKING:
    from baybe.searchspace.core import SearchSpace


@define
class SubspaceContinuous(SerialMixin):
    """Class for managing continuous subspaces.

    Builds the subspace from parameter definitions, keeps
    track of search metadata, and provides access to candidate sets and different
    parameter views.
    """

    parameters: tuple[NumericalContinuousParameter, ...] = field(
        converter=to_tuple, validator=lambda _, __, x: validate_parameter_names(x)
    )
    """The list of parameters of the subspace."""

    constraints_lin_eq: tuple[ContinuousLinearEqualityConstraint, ...] = field(
        converter=to_tuple, factory=tuple
    )
    """List of linear equality constraints."""

    constraints_lin_ineq: tuple[
        ContinuousLinearInequalityConstraint, ...
    ] = field(converter=to_tuple, factory=tuple)
    """List of linear inequality constraints."""

    constraints_cardinality: tuple[ContinuousCardinalityConstraint, ...] = field(
        converter=to_tuple, factory=tuple, validator=lambda _, __,
                                                            x:
        validate_continuous_cardinality_constraints(x))
    """List of cardinality constraints."""

    def __str__(self) -> str:
        if self.is_empty:
            return ""

        start_bold = "\033[1m"
        end_bold = "\033[0m"

        # Convert the lists to dataFrames to be able to use pretty_printing
        param_list = [param.summary() for param in self.parameters]
        eq_constraints_list = [constr.summary() for constr in self.constraints_lin_eq]
        ineq_constraints_list = [
            constr.summary() for constr in self.constraints_lin_ineq
        ]
        param_df = pd.DataFrame(param_list)
        lin_eq_constr_df = pd.DataFrame(eq_constraints_list)
        lin_ineq_constr_df = pd.DataFrame(ineq_constraints_list)

        # Put all attributes of the continuous class in one string
        continuous_str = f"""{start_bold}Continuous Search Space{end_bold}
            \n{start_bold}Continuous Parameters{end_bold}\n{pretty_print_df(param_df)}
            \n{start_bold}List of Linear Equality Constraints{end_bold}
            \r{pretty_print_df(lin_eq_constr_df)}
            \n{start_bold}List of Linear Inequality Constraints{end_bold}
            \r{pretty_print_df(lin_ineq_constr_df)}"""

        return continuous_str.replace("\n", "\n ").replace("\r", "\r ")

    def to_searchspace(self) -> SearchSpace:
        """Turn the subspace into a search space with no discrete part."""
        from baybe.searchspace.core import SearchSpace

        return SearchSpace(continuous=self)

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
            NumericalContinuousParameter(cast(str, name), bound)
            for (name, bound) in bounds.items()
        ]
        return SubspaceContinuous(parameters)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        parameters: Sequence[ContinuousParameter] | None = None,
    ) -> SubspaceContinuous:
        """Create a hyperrectangle-shaped continuous subspace from a dataframe.

        More precisely, create the smallest axis-aligned hyperrectangle-shaped
        continuous subspace that contains the points specified in the given dataframe.

        Args:
            df: The dataframe specifying the points spanning the subspace.
            parameters: Optional parameter objects corresponding to the columns in the
                given dataframe that can be provided to explicitly control parameter
                attributes. If a match between column name and parameter name is found,
                the corresponding parameter object is used. If a column has no match in
                the parameter list, a new
                :class:`baybe.parameters.numerical.NumericalContinuousParameter`
                is created with default optional arguments. For more details, see
                :func:`baybe.parameters.utils.get_parameters_from_dataframe`.

        Raises:
            ValueError: If parameter types other than
                :class:`baybe.parameters.numerical.NumericalContinuousParameter`
                are provided.

        Returns:
            The created continuous subspace.
        """
        # TODO: Add option for convex hull once constraints are in place

        if parameters and not all(
            isinstance(p, NumericalContinuousParameter) for p in parameters
        ):
            raise ValueError(
                "Currently, only parameters of type "
                "'{NumericalContinuousParameter.__name__}' are supported."
            )

        def continuous_parameter_factory(name: str, values: Collection[Any]):
            return NumericalContinuousParameter(name, (min(values), max(values)))

        # Get the full list of both explicitly and implicitly defined parameter
        parameters = get_parameters_from_dataframe(
            df, continuous_parameter_factory, parameters
        )

        return cls(parameters)

    @property
    def is_empty(self) -> bool:
        """Return whether this subspace is empty."""
        return len(self.parameters) == 0

    @property
    def param_names(self) -> tuple[str, ...]:
        """Return list of parameter names."""
        return tuple(p.name for p in self.parameters)

    @property
    def param_bounds_comp(self) -> np.ndarray:
        """Return bounds as numpy array."""
        if not self.parameters:
            return np.empty((2, 0), dtype=DTypeFloatNumpy)
        return np.stack([p.bounds.to_ndarray() for p in self.parameters]).T

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """See :func:`baybe.searchspace.discrete.SubspaceDiscrete.transform`.

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
            A dataframe containing the points as rows with columns corresponding to the
            parameter names.
        """
        if not self.parameters:
            return pd.DataFrame()
        import torch
        from botorch.utils.sampling import get_polytope_samples

        # TODO Revisit: torch and botorch here are actually only necessary if there
        # are constraints. If there are none and the lists are empty we can just sample
        # without the get_polytope_samples, which means torch and botorch
        # wouldn't be needed.

        if len(self.constraints_cardinality) != 0:
            return self._sample_with_cardinality_constraints(n_points)
        else:
            points = get_polytope_samples(
                n=n_points,
                bounds=torch.from_numpy(self.param_bounds_comp),
                equality_constraints=[
                    c.to_botorch(self.parameters) for c in self.constraints_lin_eq
                ],
                inequality_constraints=[
                    c.to_botorch(self.parameters) for c in self.constraints_lin_ineq
                ],
            )
            return pd.DataFrame(points, columns=self.param_names)

    def _sample_with_cardinality_constraints(self, n_points: int = 1) -> pd.DataFrame:
        """Get random samples from the continuous space with cardinality constraints.

        Args:
            n_points: see sample_random()

        Returns:
            see sample_random()
        """
        import torch
        from botorch.utils.sampling import get_polytope_samples
        # TODO Revisit: torch and botorch here are actually only necessary if there
        # are constraints. If there are none and the lists are empty we can just sample
        # without the get_polytope_samples, which means torch and botorch
        # wouldn't be needed. see samples_random().

        if len(self.constraints_cardinality) != 0:
            # generate all possibilities of inactive parameters which are compatible
            # to all cardinality constraints
            _, inactive_params_full_list = get_combinations_from_cardinality_con(
                list(self.constraints_cardinality)
            )

            points_all = torch.Tensor()
            i_ite = 0
            n_ite_thres = 1e5
            while len(points_all) < n_points:
                # randomly choose one set of inactive parameters
                inactive_params_sampled = random.choice(inactive_params_full_list)
                bounds = self.param_bounds_comp

                # set each inactive parameter to zero by changing bounds to [0, 0]
                if len(inactive_params_sampled) != 0:
                    for idx, param in enumerate(self.parameters):
                        if param.name in inactive_params_sampled:
                            bounds[:, idx] = [0, 0]
                try:
                    point = get_polytope_samples(
                        n=1,
                        bounds=torch.from_numpy(bounds),
                        equality_constraints=[
                            c.to_botorch(self.parameters)
                            for c in self.constraints_lin_eq
                        ],
                        inequality_constraints=[
                            c.to_botorch(self.parameters)
                            for c in self.constraints_lin_ineq
                        ],
                    )

                    points_all = torch.cat((points_all, point), 0)
                except ValueError as ve:
                    print(
                        f"Caught a ValueError: with parameters"
                        f" {inactive_params_sampled} being set to zeros, {ve}"
                    )

                # avoid infinite loop
                i_ite += 1
                assert i_ite < n_ite_thres, (
                    "We are exceeding the Iteration limit "
                    "when generating random samples."
                )

            return pd.DataFrame(points_all, columns=self.param_names)

    def samples_full_factorial(self, n_points: int = 1) -> pd.DataFrame:
        """Get random point samples from the full factorial of the continuous space.

        Args:
            n_points: Number of points that should be sampled.

        Returns:
            A dataframe containing the points as rows with columns corresponding to the
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


# Among all possible active/inactive parameter setting, active parameters cannot be
# empty; while inactive parameters can be empty!
CombinationFromCardinalityConstratins: TypeAlias = tuple[
    list[list[str]], Optional[list[list[str]]]
]


def get_combinations_from_cardinality_con(
    cardinality_constraint_list: list[ContinuousCardinalityConstraint]
) -> CombinationFromCardinalityConstratins:
    """Get all possible combination of active/inactive features.

    Here, we generate a list of all possibilities, with each possibility being active
    parameters and inactive parameters, when all cardinality constraints are taken
    into account.

    Args:
        cardinality_constraint_list: a list of continuous cardinality constraints

    Returns:
        a list of lists, each list being one possibility of active parameters
        a list of lists, each list being one possibility of inactive parameters

    """
    import itertools

    assert len(cardinality_constraint_list) != 0, "There is no cardinality constraints."

    active_params_list_all = []

    # loops through each NChooseK constraint
    for con in cardinality_constraint_list:
        assert isinstance(con, ContinuousCardinalityConstraint)
        active_params_list = []
        for n in range(con.cardinality_low, con.cardinality_up + 1):
            active_params_list.extend(itertools.combinations(con.parameters, n))
        active_params_list_all.append(active_params_list)

    # product between NChooseK constraints
    active_params_list_all = list(itertools.product(*active_params_list_all))

    # format into a list of used features
    active_params_list_formatted = []
    for active_params_list in active_params_list_all:
        active_params_list_flattened = [
            item for sublist in active_params_list for item in sublist
        ]
        active_params_list_formatted.append(list(set(active_params_list_flattened)))

    # sort lists
    active_params_list_sorted = []
    for active_params in active_params_list_formatted:
        active_params_list_sorted.append(sorted(active_params))

    # drop duplicates
    active_params_list_no_dup = []
    for active_params in active_params_list_sorted:
        if active_params not in active_params_list_no_dup:
            active_params_list_no_dup.append(active_params)

    # remove combinations not fulfilling constraints
    active_params_list_final = []
    for combo in active_params_list_no_dup:
        fulfill_constraints = []  # list of bools tracking if constraints are fulfilled
        for con in cardinality_constraint_list:
            assert isinstance(con, ContinuousCardinalityConstraint)
            count = 0  # count of features in combo that are in con.features
            for param in combo:
                if param in con.parameters:
                    count += 1
            if con.cardinality_low <= count <= con.cardinality_up:
                fulfill_constraints.append(True)
            else:
                fulfill_constraints.append(False)
        if np.all(fulfill_constraints):
            active_params_list_final.append(combo)

    # generate the union of parameters that are present in cardinality constraints
    params_in_cardinality_cons = []
    for con in cardinality_constraint_list:
        assert isinstance(con, ContinuousCardinalityConstraint)
        params_in_cardinality_cons.extend(con.parameters)
    params_in_cardinality_cons = list(set(params_in_cardinality_cons))
    params_in_cardinality_cons.sort()

    # inactive parameters
    inactive_params_list = []
    for active_params in active_params_list_final:
        inactive_params_list.append(
            [
                f_key
                for f_key in params_in_cardinality_cons
                if f_key not in active_params
            ]
        )
    return active_params_list_final, inactive_params_list


# Register deserialization hook
converter.register_structure_hook(SubspaceContinuous, select_constructor_hook)
