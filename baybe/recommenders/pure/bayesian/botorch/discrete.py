"""Discrete recommendation routines for BotorchRecommender."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
from attrs import evolve

from baybe.searchspace import SubspaceDiscrete
from baybe.utils.dataframe import to_tensor

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.recommenders.pure.bayesian.botorch.core import BotorchRecommender


def recommend_discrete_with_subsets(
    recommender: BotorchRecommender,
    subspace_discrete: SubspaceDiscrete,
    batch_size: int,
) -> pd.DataFrame:
    """Recommend from a discrete space with subset-generating constraints.

    Splits the candidate set into subsets according to subset-generating constraints,
    runs optimization on each feasible subset, and returns the batch with
    the highest joint acquisition value. Subsets with fewer candidates
    than ``batch_size`` are skipped.

    Args:
        recommender: The recommender instance.
        subspace_discrete: The discrete subspace from which to generate
            recommendations.
        batch_size: The size of the recommendation batch.

    Returns:
        A dataframe containing the recommendations as a subset of rows from the
        provided experimental representation.
    """
    import torch

    candidates = subspace_discrete.get_candidates()
    masks: Iterable[npt.NDArray[np.bool_]]
    if subspace_discrete.n_subsets <= recommender.max_n_subsets:
        masks = subspace_discrete.subset_masks(min_candidates=batch_size)
    else:
        masks = subspace_discrete.sample_subset_masks(
            recommender.max_n_subsets,
            min_candidates=batch_size,
        )

    def make_callable(
        mask: np.ndarray,
    ) -> Callable[[], tuple[pd.DataFrame, Tensor]]:
        def optimize() -> tuple[pd.DataFrame, Tensor]:
            subset_subspace = evolve(subspace_discrete, exp_rep=candidates.loc[mask])

            rec = recommend_discrete_without_subsets(
                recommender, subset_subspace, batch_size
            )

            comp = subspace_discrete.transform(rec)
            with torch.no_grad():
                acqf_value = recommender._botorch_acqf(to_tensor(comp).unsqueeze(0))
            return rec, acqf_value

        return optimize

    callables = (make_callable(m) for m in masks)
    best_rec, _ = recommender._optimize_over_subsets(callables)
    return best_rec


def recommend_discrete_without_subsets(
    recommender: BotorchRecommender,
    subspace_discrete: SubspaceDiscrete,
    batch_size: int,
) -> pd.DataFrame:
    """Generate recommendations from a discrete search space.

    Args:
        recommender: The recommender instance.
        subspace_discrete: The discrete subspace from which to generate
            recommendations.
        batch_size: The size of the recommendation batch.

    Raises:
        IncompatibleAcquisitionFunctionError: If a non-Monte Carlo acquisition
            function is used with a batch size > 1.

    Returns:
        A dataframe containing the recommendations as a subset of rows from the
        provided experimental representation.
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

    # Determine the next set of points to be tested
    candidates_comp = subspace_discrete.transform(subspace_discrete.get_candidates())
    points, _ = optimize_acqf_discrete(
        recommender._botorch_acqf, batch_size, to_tensor(candidates_comp)
    )

    # Retrieve the rows from the subspace corresponding to the selected points
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

    return subspace_discrete.get_candidates().loc[idxs]
