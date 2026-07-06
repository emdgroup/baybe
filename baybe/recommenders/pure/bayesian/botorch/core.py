"""Deprecated!"""

from __future__ import annotations

import warnings
from typing import Any

from baybe.acquisition.base import AcquisitionFunction
from baybe.optimizers import ContinuousOptimizer
from baybe.recommenders.pure.base import PureRecommender
from baybe.recommenders.pure.bayesian.core import BayesianRecommender
from baybe.searchspace import SearchSpaceType
from baybe.serialization.core import converter
from baybe.surrogates.base import SurrogateProtocol
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.utils.sampling_algorithms import DiscreteSamplingMethod

# >>>>>>>>>> Deprecation


def _structure_botorch_recommender(
    val: dict[str, Any], cls: type
) -> BayesianRecommender:
    """Structure hook that deserializes legacy ``BotorchRecommender`` dicts.

    Reconstructs the legacy keyword arguments from the dict and delegates to the
    :func:`BotorchRecommender` shim, which issues the deprecation warning and maps
    everything to :class:`~baybe.recommenders.pure.bayesian.core.BayesianRecommender`.
    """
    # Structure typed nested fields; everything else is forwarded as-is
    if "surrogate_model" in val:
        val["surrogate_model"] = converter.structure(
            val["surrogate_model"], SurrogateProtocol
        )
    if "acquisition_function" in val:
        val["acquisition_function"] = converter.structure(
            val["acquisition_function"], AcquisitionFunction
        )

    return BotorchRecommender(**val, _stacklevel=5)


_existing_pure_recommender_hook = converter.get_structure_hook(PureRecommender)


@converter.register_structure_hook
def _structure_pure_recommender_with_botorch_compat(
    val: dict[str, Any] | str, cls: type[PureRecommender]
) -> PureRecommender:
    """Wrap the PureRecommender structure hook to redirect legacy BotorchRecommender."""
    if isinstance(val, dict) and val.get("type") == "BotorchRecommender":
        val = dict(val)
        val.pop("type")
        return _structure_botorch_recommender(val, cls)
    return _existing_pure_recommender_hook(val, cls)


# <<<<<<<<<< Deprecation


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
    _stacklevel: int = 2,
) -> BayesianRecommender:
    """Deprecated! Use :class:`~baybe.recommenders.pure.bayesian.core.BayesianRecommender` instead."""  # noqa
    warnings.warn(
        f"'BotorchRecommender' is deprecated and will be removed in a future version. "
        f"Please use '{BayesianRecommender.__name__}' instead.",
        DeprecationWarning,
        stacklevel=_stacklevel,
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
