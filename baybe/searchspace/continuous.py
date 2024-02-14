"""Continuous subspaces."""

from __future__ import annotations

from typing import Any, Collection, List, Optional

import numpy as np
import pandas as pd
import torch
from attr import define, field

from baybe.constraints import (
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
)
from baybe.parameters import NumericalContinuousParameter
from baybe.parameters.utils import get_parameters_from_dataframe
from baybe.searchspace.discrete import parameter_cartesian_prod_to_df
from baybe.searchspace.validation import validate_parameter_names
from baybe.serialization import SerialMixin, converter, select_constructor_hook
from baybe.utils.dataframe import pretty_print_dataframe
from baybe.utils.numerical import DTypeFloatTorch


@define
class SubspaceContinuous(SerialMixin):
    """Class for managing continuous subspaces.

    Builds the subspace from parameter definitions, keeps
    track of search metadata, and provides access to candidate sets and different
    parameter views.
    """

    parameters: List[NumericalContinuousParameter] = field(
        validator=lambda _1, _2, x: validate_parameter_names(x)
    )
    """The list of parameters of the subspace."""

    constraints_lin_eq: List[ContinuousLinearEqualityConstraint] = field(factory=list)
    """List of linear equality constraints."""

    constraints_lin_ineq: List[ContinuousLinearInequalityConstraint] = field(
        factory=list
    )
    """List of linear inequality constraints."""

    def __str__(self) -> str:
        if self.is_empty:
            return ""
        # Convert the lists to dataFrames to be able to use pretty_printing
        par_df = parameter_cartesian_prod_to_df(self.parameters)
        const_lin_eq_df = parameter_cartesian_prod_to_df(self.constraints_lin_eq)
        const_lin_ineq_df = parameter_cartesian_prod_to_df(self.constraints_lin_ineq)

        # Put all attributes of the continuous class in one string
        continuous_str = (
            "\n\n \033[1m |--> The continuous search space \n\n \033[0m"
            + "\033[1m Continuous Parameters \033[0m"
            + " \n"
            + pretty_print_dataframe(par_df)
            + "\n"
            + "\n "
            + "\033[1m List of linear equality constraints \033[0m"
            + " \n"
            + pretty_print_dataframe(const_lin_eq_df)
            + "\n"
            + "\n"
            + "\033[1m List of linear inequality constraints \033[0m"
            + " \n"
            + pretty_print_dataframe(const_lin_ineq_df)
            + "\n\n"
        )

        return continuous_str

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
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        parameters: Optional[List[NumericalContinuousParameter]] = None,
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

        Returns:
            The created continuous subspace.
        """
        # TODO: Add option for convex hull once constraints are in place

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
            A data frame containing the points as rows with columns corresponding to the
            parameter names.
        """
        if not self.parameters:
            return pd.DataFrame()

        from botorch.utils.sampling import get_polytope_samples

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


# Register deserialization hook
converter.register_structure_hook(SubspaceContinuous, select_constructor_hook)
