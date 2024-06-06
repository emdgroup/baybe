"""Continuous constraints."""


import math

import numpy as np
from attrs import define, field
from attrs.validators import ge, instance_of

from baybe.constraints.base import ContinuousConstraint, ContinuousLinearConstraint


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
class ContinuousCardinalityConstraint(ContinuousConstraint):
    """Class for continuous cardinality constraints.

    The constraint is defined as ``cardinality[..., x_i, ...] >= min_cardinality´´
    and ``cardinality[..., x_i, ...] <= max_cardinality´´.
    """

    min_cardinality: int = field(default=0, validator=[instance_of(int), ge(0)])
    "The lower limit of cardinality"

    max_cardinality: int = field(validator=instance_of(int))
    "The upper limit of cardinality"

    @max_cardinality.default
    def _default_max_cardinality(self):
        """Set max_cardinality = len(parameter) by default."""
        return len(self.parameters)

    def __attrs_post_init__(self):
        """Validate."""
        if self.min_cardinality > self.max_cardinality:
            raise ValueError(
                f"The lower bound of cardinality should not be larger "
                f"than the upper bound, but was upper bound = "
                f"{self.max_cardinality}, lower bound ="
                f" {self.min_cardinality}."
            )

        if self.max_cardinality > len(self.parameters):
            raise ValueError(
                f"The upper bound of cardinality should not exceed the "
                f"number of parameters, but was upper bound ="
                f" {self.max_cardinality}, and len(parameters) ="
                f" {len(self.parameters)}."
            )

        if self.min_cardinality == 0 and self.max_cardinality == len(self.parameters):
            raise ValueError(
                "No cardinality constraint is required when "
                "0<= cardinality <= len(parameters)."
            )

    def sample_inactive_parameters(self, batch_size: int = 1) -> list[set[str]]:
        """Generate inactive parameters based on cardinality constraints.

        Args:
            batch_size: Number of parameter configurations that should be sampled.

        Returns:
            a list of samples with each sample being a collection of inactive
            parameters names.
        """
        # combinatorial for each cardinality
        n_configurations_per_cardinality = [
            math.comb(len(self.parameters), n)
            for n in range(self.min_cardinality, self.max_cardinality + 1)
        ]

        # probability of each cardinality
        probabilities = n_configurations_per_cardinality / np.sum(
            n_configurations_per_cardinality
        )

        # randomly generate #(inactive parameters)
        n_active_params = np.random.choice(
            np.arange(self.min_cardinality, self.max_cardinality + 1),
            batch_size,
            p=probabilities,
        )
        n_inactive_params = len(self.parameters) - n_active_params

        # sample inactive parameters
        inactive_params = [
            set(np.random.choice(self.parameters, n_inactive, replace=False))
            for n_inactive in n_inactive_params
        ]

        return inactive_params
