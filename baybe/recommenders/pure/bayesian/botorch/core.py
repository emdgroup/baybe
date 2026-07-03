"""Deprecated!."""

from __future__ import annotations

import copy
import warnings
from typing import Any

from cattrs.gen import make_dict_structure_fn

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
# Field name remappings from legacy BotorchRecommender into the optimizer sub-dict
_OPTIMIZER_FIELD_RENAMES = {
    "n_restarts": "n_starts",
    "n_raw_samples": "n_initial_samples",
    "sequential_continuous": "sequential",
}


def _structure_botorch_recommender(
    val: dict[str, Any], cls: type
) -> BayesianRecommender:
    """Structure hook that deserializes legacy ``BotorchRecommender`` dicts."""
    warnings.warn(
        f"'BotorchRecommender' is deprecated and will be removed in a future version. "
        f"Please use '{BayesianRecommender.__name__}' instead.",
        DeprecationWarning,
        stacklevel=4,
    )

    val = copy.deepcopy(val)  # copy to avoid mutating caller's dict

    # Extract and rename any legacy optimizer fields that are explicitly present
    optimizer_kwargs: dict[str, Any] = {}
    for legacy_name, new_name in _OPTIMIZER_FIELD_RENAMES.items():
        if legacy_name in val:
            optimizer_kwargs[new_name] = val.pop(legacy_name)

    # Drop legacy fields not yet supported on BayesianRecommender
    for unsupported in ("hybrid_sampler", "sampling_percentage", "max_n_subsets"):
        val.pop(unsupported, None)

    val["optimizer"] = {"type": ContinuousOptimizer.__name__, **optimizer_kwargs}

    return make_dict_structure_fn(BayesianRecommender, converter)(
        val, BayesianRecommender
    )


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
