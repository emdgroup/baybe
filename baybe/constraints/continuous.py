"""Continuous constraints."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from attrs import define

from baybe.constraints.base import (
    CardinalityConstraint,
    ContinuousLinearConstraint,
    ContinuousNonlinearConstraint,
)
from baybe.parameters import NumericalContinuousParameter

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.constraints.base import FuncNonlinearInequality

# Boolean variable indicating intra-/inter-point constraints used in botorch.
INTRA_POINT = True


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

    def to_botorch(  # noqa: D102
        self, parameters: Sequence[NumericalContinuousParameter], idx_offset: int = 0
    ) -> list[tuple[FuncNonlinearInequality, bool]]:
        # See base class.

        import torch

        from baybe.utils.cardinality_constraint import (
            max_cardinality_relaxed,
            min_cardinality_relaxed,
        )

        def chain_callable_with_parameter_selection(
            _func_cardinality_relaxed: FuncNonlinearInequality, _indices: Tensor
        ) -> FuncNonlinearInequality:
            """Augment a parameter selection operation to the callable.

            Args:
                _func_cardinality_relaxed: A callable that evaluates the relaxed
                    cardinality constraint.
                _indices: Indices of constraint parameters.

            Returns:
                A callable that chains parameter selection to the input callable.
            """

            def callable_at_broader_parameters(x: Tensor) -> Tensor:
                """Apply a callable to the specified parameters."""
                return _func_cardinality_relaxed(x[..., _indices])

            return callable_at_broader_parameters

        # Relaxed cardinality constraint callable over constraint parameters
        func_relaxed_cardinality_on_constraint_params = []
        if self.max_cardinality != len(self.parameters):
            func_relaxed_cardinality_on_constraint_params.append(
                partial(max_cardinality_relaxed, self.max_cardinality)
            )
        if self.min_cardinality != 0:
            func_relaxed_cardinality_on_constraint_params.append(
                partial(min_cardinality_relaxed, self.min_cardinality)
            )

        # Get indices of constraint parameters in the whole search space
        param_names = [p.name for p in parameters]
        indices = torch.tensor(
            [
                param_names.index(key) + idx_offset
                for key in self.parameters
                if key in param_names
            ]
        )

        # Relaxed cardinality constraint callable over search space parameters
        func_relaxed_cardinality_on_searchspace = [
            chain_callable_with_parameter_selection(func, indices)
            for func in func_relaxed_cardinality_on_constraint_params
        ]

        return [(func, INTRA_POINT) for func in func_relaxed_cardinality_on_searchspace]
