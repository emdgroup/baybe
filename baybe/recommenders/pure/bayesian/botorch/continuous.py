"""Continuous recommendation routines for BotorchRecommender."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Iterable
from typing import TYPE_CHECKING

import pandas as pd
from attrs import fields

from baybe.constraints.utils import is_cardinality_fulfilled
from baybe.exceptions import (
    IncompatibilityError,
    MinimumCardinalityViolatedWarning,
)
from baybe.parameters.numerical import _FixedNumericalContinuousParameter
from baybe.searchspace import SubspaceContinuous
from baybe.utils.basic import flatten

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.recommenders.pure.bayesian.botorch.core import BotorchRecommender


def recommend_continuous_torch(
    recommender: BotorchRecommender,
    subspace_continuous: SubspaceContinuous,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Dispatcher selecting the continuous optimization routine."""
    if subspace_continuous.n_subsets > 0:
        return recommend_continuous_with_cardinality_constraints(
            recommender, subspace_continuous, batch_size
        )
    else:
        return recommend_continuous_without_cardinality_constraints(
            recommender, subspace_continuous, batch_size
        )


def recommend_continuous_with_cardinality_constraints(
    recommender: BotorchRecommender,
    subspace_continuous: SubspaceContinuous,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Recommend from a continuous space with cardinality constraints.

    Optimizes the acquisition function across subsets defined by cardinality
    constraints and returns the best result.

    The specific collection of subsets considered by the recommender is obtained
    as either the full combinatorial set of possible parameter splits or a random
    selection thereof, depending on the upper bound specified by the corresponding
    recommender attribute.

    In each subset, the constraint-imposed configuration is fixed, so that the
    constraints can be removed and a regular optimization can be performed. The
    recommendation is then constructed from the combined optimization results of the
    unconstrained spaces.

    Args:
        recommender: The recommender instance.
        subspace_continuous: The continuous subspace from which to generate
            recommendations.
        batch_size: The size of the recommendation batch.

    Returns:
        The recommendations and corresponding acquisition values.

    Raises:
        ValueError: If the continuous search space has no cardinality
            constraints.
    """
    if subspace_continuous.n_subsets == 0:
        raise ValueError(
            f"'{recommend_continuous_with_cardinality_constraints.__name__}' "
            f"expects a subspace with cardinality constraints."
        )

    # Determine search scope based on number of subset configurations
    configs: Iterable[frozenset[str]]
    if subspace_continuous.n_subsets <= recommender.max_n_subsets:
        configs = subspace_continuous.inactive_parameter_combinations()
    else:
        configs = subspace_continuous._sample_inactive_parameters(
            recommender.max_n_subsets
        )

    # Create closures for each subset configuration
    def make_callable(
        inactive_params: Collection[str],
    ) -> Callable[[], tuple[Tensor, Tensor]]:
        def optimize() -> tuple[Tensor, Tensor]:
            import torch

            sub = subspace_continuous._enforce_cardinality_constraints(inactive_params)
            # Note: We explicitly evaluate the acqf function for the batch
            # because the object returned by the optimization routine may
            # contain joint or individual acquisition values, depending on
            # whether sequential or joint optimization is applied
            p, _ = recommend_continuous_torch(recommender, sub, batch_size)
            with torch.no_grad():
                acqf_value = recommender._botorch_acqf(p)
            return p, acqf_value

        return optimize

    callables = (make_callable(ip) for ip in configs)
    points, acqf_value = recommender._optimize_over_subsets(callables)

    # Check if any minimum cardinality constraints are violated
    if not is_cardinality_fulfilled(
        pd.DataFrame(points, columns=subspace_continuous.parameter_names),
        subspace_continuous,
        check_maximum=False,
    ):
        warnings.warn(
            "At least one minimum cardinality constraint has been violated. "
            "This may occur when parameter ranges extend beyond zero in both "
            "directions, making the feasible region non-convex. For such "
            "parameters, minimum cardinality constraints are currently not "
            "enforced due to the complexity of the resulting optimization problem.",
            MinimumCardinalityViolatedWarning,
        )

    return points, acqf_value


def recommend_continuous_without_cardinality_constraints(
    recommender: BotorchRecommender,
    subspace_continuous: SubspaceContinuous,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Recommend from a continuous search space without cardinality constraints.

    Args:
        recommender: The recommender instance.
        subspace_continuous: The continuous subspace from which to generate
            recommendations.
        batch_size: The size of the recommendation batch.

    Returns:
        The recommendations and corresponding acquisition values.

    Raises:
        ValueError: If the continuous search space has cardinality constraints.
    """
    import torch
    from botorch.optim import optimize_acqf

    if subspace_continuous.n_subsets > 0:
        raise ValueError(
            f"'{recommend_continuous_without_cardinality_constraints.__name__}' "
            f"expects a subspace without cardinality constraints."
        )

    fixed_parameters = {
        idx: p.value
        for (idx, p) in enumerate(subspace_continuous.parameters)
        if isinstance(p, _FixedNumericalContinuousParameter)
    }

    # TODO: Add option for automatic choice once the "settings" PR is merged,
    #   which ships the necessary machinery
    if (
        recommender.sequential_continuous
        and subspace_continuous.has_interpoint_constraints
    ):
        from baybe.recommenders.pure.bayesian.botorch.core import BotorchRecommender

        raise IncompatibilityError(
            f"Setting the "
            f"'{fields(BotorchRecommender).sequential_continuous.name}' "
            f"flag to ``True`` while interpoint constraints are present in the "
            f"continuous subspace is not supported. "
        )

    # NOTE: The explicit `or None` conversion is added as an additional safety net
    #   because it is unclear if the corresponding presence checks for these
    #   arguments is correctly implemented in all invoked BoTorch subroutines.
    #   For details: https://github.com/pytorch/botorch/issues/2042
    points, acqf_values = optimize_acqf(
        acq_function=recommender._botorch_acqf,
        bounds=torch.from_numpy(subspace_continuous.comp_rep_bounds.values),
        q=batch_size,
        num_restarts=recommender.n_restarts,
        raw_samples=recommender.n_raw_samples,
        fixed_features=fixed_parameters or None,
        equality_constraints=flatten(
            c.to_botorch(
                subspace_continuous.parameters,
                batch_size=batch_size if c.is_interpoint else None,
            )
            for c in subspace_continuous.constraints_lin_eq
        )
        or None,
        inequality_constraints=flatten(
            c.to_botorch(
                subspace_continuous.parameters,
                batch_size=batch_size if c.is_interpoint else None,
            )
            for c in subspace_continuous.constraints_lin_ineq
        )
        or None,
        sequential=recommender.sequential_continuous,
    )
    return points, acqf_values
