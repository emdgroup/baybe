"""Continuous subspaces."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch
from attr import define, field
from botorch.utils.sampling import get_polytope_samples

from baybe.constraints import (
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
)
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace.validation import validate_parameter_names
from baybe.utils import DTypeFloatTorch


@define
class SubspaceContinuous:
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
            The constructed subspace.
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
