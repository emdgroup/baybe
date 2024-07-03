"""Continuous constraints."""

import math
from itertools import combinations
from math import comb

import numpy as np
from attrs import define

from baybe.constraints.base import (
    CardinalityConstraint,
    ContinuousLinearConstraint,
    ContinuousNonlinearConstraint,
)


@define
class ContinuousLinearEqualityConstraint(ContinuousLinearConstraint):
    """Class for continuous equality constraints.

    The constraint is defined as ``sum_i[ x_i * c_i ] == rhs``, where x_i are the
    parameter names from ``parameters`` and c_i are the entries of ``coefficients``.
    The constraint is typically fulfilled up to a small numerical tolerance.

    The class has no real content as it only serves the purpose of distinguishing the
    constraints.
    """


@define
class ContinuousLinearInequalityConstraint(ContinuousLinearConstraint):
    """Class for continuous inequality constraints.

    The constraint is defined as ``sum_i[ x_i * c_i ] >= rhs``, where x_i are the
    parameter names from ``parameters`` and c_i are the entries of ``coefficients``.
    If you want to implement a constraint of the form `<=`, multiply ``rhs`` and
    ``coefficients`` by -1. The constraint is typically fulfilled up to a small
    numerical tolerance.

    The class has no real content as it only serves the purpose of
    distinguishing the constraints.
    """


@define
class ContinuousCardinalityConstraint(
    CardinalityConstraint, ContinuousNonlinearConstraint
):
    """Class for continuous cardinality constraints."""

    @property
    def combinatorial_counts_zero_parameters(self) -> int:
        """Return the total number of all possible combinations of zero parameters."""
        combinatorial_counts = 0
        for i_zeros in range(
            len(self.parameters) - self.max_cardinality,
            len(self.parameters) - self.min_cardinality + 1,
        ):
            combinatorial_counts += comb(len(self.parameters), i_zeros)
        return combinatorial_counts

    @property
    def combinatorial_zero_parameters(self) -> list[tuple[str, ...]]:
        """Return a combinatorial list of all possible zero parameters."""
        combinatorial_zeros = []
        for i_zeros in range(
            len(self.parameters) - self.max_cardinality,
            len(self.parameters) - self.min_cardinality + 1,
        ):
            combinatorial_zeros.extend(combinations(self.parameters, i_zeros))
        return combinatorial_zeros

    def sample_inactive_parameters(self, batch_size: int = 1) -> list[set[str]]:
        """Sample sets of inactive parameters according to the cardinality constraints.

        Args:
            batch_size: The number of parameter sets to be sampled.

        Returns:
            A list of sampled inactive parameter sets, where each set holds the
            corresponding parameter names.
        """
        # The number of possible parameter configuration per set cardinality
        n_configurations_per_cardinality = [
            math.comb(len(self.parameters), n)
            for n in range(self.min_cardinality, self.max_cardinality + 1)
        ]

        # Probability of each set cardinality under the assumption that all possible
        # inactive parameter sets are equally likely
        probabilities = n_configurations_per_cardinality / np.sum(
            n_configurations_per_cardinality
        )

        # Sample the number of active/inactive parameters
        n_active_params = np.random.choice(
            np.arange(self.min_cardinality, self.max_cardinality + 1),
            batch_size,
            p=probabilities,
        )
        n_inactive_params = len(self.parameters) - n_active_params

        # Sample the inactive parameters
        inactive_params = [
            set(np.random.choice(self.parameters, n_inactive, replace=False))
            for n_inactive in n_inactive_params
        ]

        return inactive_params
