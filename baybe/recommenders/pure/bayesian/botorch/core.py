"""Botorch recommender core."""

from __future__ import annotations

import gc
import warnings
from typing import TYPE_CHECKING

from baybe.acquisition.base import AcquisitionFunction
from baybe.optimizers import GradientOptimizer
from baybe.searchspace import SearchSpaceType
from baybe.surrogates.base import SurrogateProtocol
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.utils.sampling_algorithms import DiscreteSamplingMethod

if TYPE_CHECKING:
    from baybe.recommenders.pure.bayesian.core import BayesianRecommender


def BotorchRecommender(
    *,
    surrogate_model: SurrogateProtocol = GaussianProcessSurrogate(),
    acquisition_function: AcquisitionFunction | None = None,
    sequential_continuous: bool = True,
    hybrid_sampler: DiscreteSamplingMethod | None = None,
    sampling_percentage: float = 1.0,
    n_restarts: int = 10,
    n_raw_samples: int = 64,
    max_n_subsets: int = 10,
    max_n_subspaces: int | None = None,
) -> BayesianRecommender:
    """Use factory function for BotorchRecommender deprecation.

    This recommender will be deprecated in a future version.
    This function provides the interface for creating a BayesianRecommender
    based on a BotorchRecommender.

    Args:
        surrogate_model: The surrogate model to be used.
        acquisition_function: The acquisition function to be used.
        sequential_continuous: See :class:`BayesianRecommender`.
        hybrid_sampler: See :class:`BayesianRecommender`.
        sampling_percentage: See :class:`BayesianRecommender`.
        n_restarts: See :class:`BayesianRecommender`.
        n_raw_samples: See :class:`BayesianRecommender`.
        max_n_subsets: See :class:`BayesianRecommender`.
        max_n_subspaces: Deprecated! Use ``max_n_subsets`` instead.

    Returns:
        BayesianRecommender: Instance of `BayesianRecommender` with provided parameters.
    """
    from baybe.recommenders.pure.bayesian.core import BayesianRecommender

    if max_n_subspaces is not None:
        warnings.warn(
            "'max_n_subspaces' has been renamed to 'max_n_subsets' and will "
            "be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        max_n_subsets = max_n_subspaces

    warnings.warn(
        "'BotorchRecommender' is deprecated and will be removed in a future version. "
        "Please use 'BayesianRecommender' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # TODO: Clean up once more optimizers are implemented.
    if not sequential_continuous:
        return BayesianRecommender(
            surrogate_model=surrogate_model,
            acquisition_function=acquisition_function,
            optimizer=GradientOptimizer(
                n_restarts=n_restarts,
                n_raw_samples=n_raw_samples,
                sequential_continuous=sequential_continuous,
            ),
            hybrid_sampler=hybrid_sampler,
            sampling_percentage=sampling_percentage,
            n_restarts=n_restarts,
            n_raw_samples=n_raw_samples,
            max_n_subsets=max_n_subsets,
        )
    else:
        return BayesianRecommender(
            surrogate_model=surrogate_model,
            acquisition_function=acquisition_function,
            hybrid_sampler=hybrid_sampler,
            sampling_percentage=sampling_percentage,
            n_restarts=n_restarts,
            n_raw_samples=n_raw_samples,
            max_n_subsets=max_n_subsets,
        )


BotorchRecommender.compatibility = SearchSpaceType.HYBRID  # type: ignore[attr-defined]
BotorchRecommender.supports_discrete_subset_generating_constraints = True  # type: ignore[attr-defined]
# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
