"""Continuous subspaces."""

from __future__ import annotations

import warnings
from collections.abc import Collection, Container, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from attr import define, field

from baybe.constraints import (
    ContinuousCardinalityConstraint,
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
)
from baybe.constraints.validation import validate_continuous_cardinality_constraints
from baybe.exceptions import SamplingFailedError
from baybe.parameters import NumericalContinuousParameter
from baybe.parameters.base import ContinuousParameter
from baybe.parameters.utils import get_parameters_from_dataframe
from baybe.searchspace.validation import validate_parameter_names
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

    constraints_lin_ineq: tuple[ContinuousLinearInequalityConstraint, ...] = field(
        converter=to_tuple, factory=tuple
    )
    """List of linear inequality constraints."""

    constraints_cardinality: tuple[ContinuousCardinalityConstraint, ...] = field(
        converter=to_tuple,
        factory=tuple,
        validator=lambda _, __, x: validate_continuous_cardinality_constraints(x),
    )
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
        cardinality_constraints_list = [
            constr.summary() for constr in self.constraints_cardinality
        ]
        param_df = pd.DataFrame(param_list)
        lin_eq_constr_df = pd.DataFrame(eq_constraints_list)
        lin_ineq_constr_df = pd.DataFrame(ineq_constraints_list)
        cardinality_constr_df = pd.DataFrame(cardinality_constraints_list)

        # Put all attributes of the continuous class in one string
        continuous_str = f"""{start_bold}Continuous Search Space{end_bold}
            \n{start_bold}Continuous Parameters{end_bold}\n{pretty_print_df(param_df)}
            \n{start_bold}List of Linear Equality Constraints{end_bold}
            \r{pretty_print_df(lin_eq_constr_df)}
            \n{start_bold}List of Linear Inequality Constraints{end_bold}
            \r{pretty_print_df(lin_ineq_constr_df)}
            \n{start_bold}List of Cardinality Constraints{end_bold}
            \r{pretty_print_df(cardinality_constr_df)}"""

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

    def samples_random(
        self, n_points: int = 1, bounds: np.ndarray | None = None
    ) -> pd.DataFrame:
        """Get random point samples from the continuous space.

        Notes:
            Instead of using self.param_bounds_comp, we use the input "bounds" to
            indicate the parameter bounds. This is because we need to set the
            bounds of an inactive parameter to [0, 0], which is however not allowed
            by Interval.
            TODO: if Interval allows lower==upper, no need to keep the additional
            bounds.

        Args:
            n_points: Number of points that should be sampled.
            bounds: Parameter bounds. Note that the bounds here may differ from that
                contained in self.parameters, e.g. bounds(inactive parameter) = [0, 0].


        Returns:
            A dataframe containing the points as rows with columns corresponding to the
            parameter names.
        """
        if not self.parameters:
            return pd.DataFrame()

        if bounds is None:
            bounds = self.param_bounds_comp

        if (
            len(self.constraints_lin_eq) == 0
            and len(self.constraints_lin_ineq) == 0
            and len(self.constraints_cardinality) == 0
        ):
            return self._sample_from_bounds(n_points, bounds)
        elif len(self.constraints_cardinality) == 0:
            return self._sample_from_polytope(n_points, bounds)
        else:
            return self._sample_with_cardinality_constraints(n_points)

    def _sample_from_bounds(self, n_points: int, bounds: np.ndarray) -> pd.DataFrame:
        """Get random samples over space without any constraints.

        Args:
            n_points: See samples_random().
            bounds: See samples_random().

        Returns:
            See samples_random().
        """
        points = np.random.uniform(
            low=bounds[0, :], high=bounds[1, :], size=(n_points, len(self.parameters))
        )

        return pd.DataFrame(points, columns=self.param_names)

    def _sample_from_polytope(self, n_points: int, bounds: np.ndarray) -> pd.DataFrame:
        """Get random samples over space with only linear constraints.

        Args:
            n_points: See samples_random().
            bounds: See samples_random().

        Returns:
            See samples_random().
        """
        import torch
        from botorch.utils.sampling import get_polytope_samples

        points = get_polytope_samples(
            n=n_points,
            bounds=torch.from_numpy(bounds),
            equality_constraints=[
                c.to_botorch(self.parameters) for c in self.constraints_lin_eq
            ],
            inequality_constraints=[
                c.to_botorch(self.parameters) for c in self.constraints_lin_ineq
            ],
        )
        return pd.DataFrame(points, columns=self.param_names)

    def _sample_with_cardinality_constraints(self, n_points: int) -> pd.DataFrame:
        """Get random samples from the continuous space with cardinality constraints.

        Args:
            n_points: See sample_random().

        Returns:
            see Sample_random().
        """
        assert (
            len(self.constraints_cardinality) != 0
        ), "No need to call this method if there is no cardinality constraint."

        points_all = pd.DataFrame(columns=[param.name for param in self.parameters])
        i_ite, N_ITE_THRES = 0, 1e5  # limit of iteration

        while points_all.shape[0] < n_points:
            # sample inactive parameters
            inactive_params_sample = self._sample_inactive_params(1)[0]

            # subspace excluding the cardinality constraints
            subspace_cardinality_cleaned = SubspaceContinuous(
                parameters=self.parameters,
                constraints_lin_eq=self.constraints_lin_eq,
                constraints_lin_ineq=self.constraints_lin_ineq,
            )

            # generate samples
            if len(inactive_params_sample):
                # set each inactive parameter to zero by changing bounds to [0, 0]
                bounds_cleaned = self._get_bounds_with_inactive_params(
                    inactive_params_sample
                )

                # sample from the subspace, in which the cardinality constraints are
                # excluded and bounds(inactive parameters) = (0, 0)
                try:
                    points_sample = subspace_cardinality_cleaned.samples_random(
                        1, bounds_cleaned
                    )
                    points_all = pd.concat((points_all, points_sample), axis=0)
                except SamplingFailedError:
                    warnings.warn(
                        f"No samples can be drawn when inactive parameters ="
                        f" {inactive_params_sample}. Try other candidates of "
                        f"inactive parameters."
                    )
            else:
                sample = subspace_cardinality_cleaned.samples_random(1)
                points_all = pd.concat((points_all, sample), axis=0)

            # avoid infinite loop
            i_ite += 1
            assert i_ite < N_ITE_THRES, (
                "We are exceeding the limit, yet we have not drawn enough samples. It "
                "appears that the feasibility area is very small. Please review the "
                "constraints."
            )

        return points_all

    def _sample_inactive_params(self, n_points: int = 1) -> list[list[str]]:
        """Sample inactive parameters randomly according to all cardinality constraints.

        Args:
            n_points: see sample_random()

        Returns:
            see sample_random()
        """
        inactive_params_samples = []
        for _ in range(n_points):
            inactive_params_samples.append(
                list(
                    set().union(
                        *(
                            con.sample_inactive_params(1)[0]
                            for con in self.constraints_cardinality
                        )
                    )
                )
            )

        return inactive_params_samples

    def _get_bounds_with_inactive_params(
        self, inactive_params: Container[str]
    ) -> np.ndarray:
        """Get parameters bounds with bounds(inactive parameters) being zeros.

        Args:
            inactive_params: names of inactive parameters

        Returns:
            bounds of parameters
        """
        bounds_cleaned = self.param_bounds_comp

        # identifying indices of inactive parameters
        inactive_param_indices = [
            idx
            for idx, param in enumerate(self.parameters)
            if param.name in inactive_params
        ]

        bounds_cleaned[:, inactive_param_indices] = 0
        return bounds_cleaned

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


# Register deserialization hook
converter.register_structure_hook(SubspaceContinuous, select_constructor_hook)
