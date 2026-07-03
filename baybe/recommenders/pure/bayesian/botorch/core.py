"""Deprecated!."""

from __future__ import annotations

import warnings

from baybe.acquisition.base import AcquisitionFunction
from baybe.optimizers import ContinuousOptimizer
from baybe.recommenders.pure.bayesian.core import BayesianRecommender
from baybe.searchspace import SearchSpaceType
from baybe.surrogates.base import SurrogateProtocol
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.utils.sampling_algorithms import DiscreteSamplingMethod


def BotorchRecommender(
    surrogate_model: SurrogateProtocol = GaussianProcessSurrogate(),
    acquisition_function: AcquisitionFunction | None = None,
    *,
    sequential_continuous: bool = True,
    hybrid_sampler: DiscreteSamplingMethod | None = None,
    sampling_percentage: float = 1.0,
    n_restarts: int = 10,
    n_raw_samples: int = 64,
    max_n_subsets: int = 10,
    max_n_subspaces: int | None = None,
) -> BayesianRecommender:
    """Deprecated! Use :class:`~baybe.recommenders.pure.bayesian.core.BayesianRecommender` instead."""  # noqa
    warnings.warn(
        f"'BotorchRecommender' is deprecated and will be removed in a future version. "
        f"Please use '{BayesianRecommender.__name__}' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if max_n_subspaces is not None:
        max_n_subsets = max_n_subspaces  # noqa: F841

    return BayesianRecommender(
        surrogate_model=surrogate_model,
        acquisition_function=acquisition_function,
        optimizer=ContinuousOptimizer(
            n_starts=n_restarts,
            n_initial_samples=n_raw_samples,
            sequential=sequential_continuous,
        ),
    )


BotorchRecommender.compatibility = SearchSpaceType.HYBRID  # type: ignore[attr-defined]
BotorchRecommender.supports_discrete_subset_generating_constraints = True  # type: ignore[attr-defined]
