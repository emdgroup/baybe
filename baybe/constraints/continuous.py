"""Continuous constraints."""

import math

import numpy as np
from attrs import define, field
from attrs.validators import ge, instance_of

from baybe.constraints.base import (
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
class ContinuousCardinalityConstraint(ContinuousNonlinearConstraint):
    """Class for continuous cardinality constraints.

    Places a constraint on the set of nonzero (i.e. "active") values among the
    specified parameters, bounding it between the two given integers,
        ``min_cardinality`` <= |{p_i : p_i != 0}| <= ``max_cardinality``
    where ``{p_i}`` are the parameters specified for the constraint.

    Note that this can be equivalently regarded as L0-constraint on the vector
    containing the specified parameters.
    """

    min_cardinality: int = field(default=0, validator=[instance_of(int), ge(0)])
    "The minimum required cardinality."

    max_cardinality: int = field(validator=instance_of(int))
    "The maximum allowed cardinality."

    @max_cardinality.default
    def _default_max_cardinality(self):
        """Use the number of involved parameters as the upper limit by default."""
        return len(self.parameters)

    def __attrs_post_init__(self):
        """Validate the cardinality bounds.

        Raises:
            ValueError: If the provided cardinality bounds are invalid.
            ValueError: If the provided cardinality bounds impose no constraint.
        """
        if self.min_cardinality > self.max_cardinality:
            raise ValueError(
                f"The lower cardinality bound cannot be larger than the upper bound. "
                f"Provided values: {self.max_cardinality=}, {self.min_cardinality=}."
            )

        if self.max_cardinality > len(self.parameters):
            raise ValueError(
                f"The cardinality bound cannot exceed the number of parameters. "
                f"Provided values: {self.max_cardinality=}, {len(self.parameters)=}."
            )

        if self.min_cardinality == 0 and self.max_cardinality == len(self.parameters):
            raise ValueError(
                f"No constraint of type `{self.__class__.__name__}' is required "
                f"when 0 <= cardinality <= len(parameters)."
            )

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
