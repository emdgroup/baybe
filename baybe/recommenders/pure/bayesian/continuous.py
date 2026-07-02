"""Continuous recommendation routines for BayesianRecommender."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Iterable
from typing import TYPE_CHECKING

import pandas as pd

from baybe.constraints.utils import is_cardinality_fulfilled
from baybe.exceptions import MinimumCardinalityViolatedWarning
from baybe.searchspace import SubspaceContinuous
from baybe.searchspace.core import SearchSpace

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.recommenders.pure.bayesian.core import BayesianRecommender


def recommend_continuous_torch(
    recommender: BayesianRecommender,
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
    recommender: BayesianRecommender,
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
    recommender: BayesianRecommender,
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
    if subspace_continuous.n_subsets > 0:
        raise ValueError(
            f"'{recommend_continuous_without_cardinality_constraints.__name__}' "
            f"expects a subspace without cardinality constraints."
        )

    points, acqf_values = recommender.optimizer(
        batch_size=batch_size,
        score_function=recommender._botorch_acqf,
        searchspace=SearchSpace(continuous=subspace_continuous),
    )
    return points, acqf_values
