"""Discrete recommendation routines for BotorchRecommender."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from baybe.searchspace import SubspaceDiscrete
from baybe.utils.dataframe import to_tensor

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.recommenders.pure.bayesian.botorch.core import BotorchRecommender


def recommend_discrete_with_partitions(
    recommender: BotorchRecommender,
    subspace_discrete: SubspaceDiscrete,
    candidates_exp: pd.DataFrame,
    batch_size: int,
) -> pd.Index:
    """Recommend from a discrete space with batch constraints.

    Partitions the candidate set according to batch constraints,
    runs optimization on each feasible partition, and returns the batch with
    the highest joint acquisition value. Partitions with fewer candidates
    than ``batch_size`` are skipped.

    Args:
        recommender: The recommender instance.
        subspace_discrete: The discrete subspace from which to generate
            recommendations.
        candidates_exp: The experimental representation of candidates.
        batch_size: The size of the recommendation batch.

    Returns:
        The dataframe indices of the recommended points.
    """
    import torch

    masks: Iterable[np.ndarray]
    if subspace_discrete.n_theoretical_partitions <= recommender.max_n_partitions:
        masks = subspace_discrete.partition_masks(
            candidates_exp, min_candidates=batch_size
        )
    else:
        masks = subspace_discrete.sample_partition_masks(
            candidates_exp, recommender.max_n_partitions, min_candidates=batch_size
        )

    def make_callable(
        mask: np.ndarray,
    ) -> Callable[[], tuple[pd.Index, Tensor]]:
        def optimize() -> tuple[pd.Index, Tensor]:
            subset = candidates_exp.loc[mask]

            idxs = recommend_discrete_without_partitions(
                recommender, subspace_discrete, subset, batch_size
            )

            comp = subspace_discrete.transform(candidates_exp.loc[idxs])
            with torch.no_grad():
                acqf_value = recommender._botorch_acqf(to_tensor(comp).unsqueeze(0))
            return idxs, acqf_value

        return optimize

    callables = (make_callable(m) for m in masks)
    best_idxs, _ = recommender._optimize_over_partitions(callables)
    return best_idxs


def recommend_discrete_without_partitions(
    recommender: BotorchRecommender,
    subspace_discrete: SubspaceDiscrete,
    candidates_exp: pd.DataFrame,
    batch_size: int,
) -> pd.Index:
    """Generate recommendations from a discrete search space.

    Args:
        recommender: The recommender instance.
        subspace_discrete: The discrete subspace from which to generate
            recommendations.
        candidates_exp: The experimental representation of all discrete candidate
            points to be considered.
        batch_size: The size of the recommendation batch.

    Raises:
        IncompatibleAcquisitionFunctionError: If a non-Monte Carlo acquisition
            function is used with a batch size > 1.

    Returns:
        The dataframe indices of the recommended points in the provided
        experimental representation.
    """
    from baybe.acquisition.acqfs import qThompsonSampling
    from baybe.exceptions import (
        IncompatibilityError,
        IncompatibleAcquisitionFunctionError,
    )

    assert recommender._objective is not None
    acqf = recommender._get_acquisition_function(recommender._objective)
    if batch_size > 1 and not acqf.supports_batching:
        raise IncompatibleAcquisitionFunctionError(
            f"The '{recommender.__class__.__name__}' only works with Monte Carlo "
            f"acquisition functions for batch sizes > 1."
        )
    if batch_size > 1 and isinstance(acqf, qThompsonSampling):
        raise IncompatibilityError(
            "Thompson sampling currently only supports a batch size of 1."
        )

    from botorch.optim import optimize_acqf_discrete

    # determine the next set of points to be tested
    candidates_comp = subspace_discrete.transform(candidates_exp)
    points, _ = optimize_acqf_discrete(
        recommender._botorch_acqf, batch_size, to_tensor(candidates_comp)
    )

    # retrieve the index of the points from the input dataframe
    # IMPROVE: The merging procedure is conceptually similar to what
    #   `SearchSpace._match_measurement_with_searchspace_indices` does, though using
    #   a simpler matching logic. When refactoring the SearchSpace class to
    #   handle continuous parameters, a corresponding utility could be extracted.
    idxs = pd.Index(
        pd.merge(
            pd.DataFrame(points, columns=candidates_comp.columns),
            candidates_comp.reset_index(),
            on=list(candidates_comp),
            how="left",
        )["index"]
    )

    return idxs
