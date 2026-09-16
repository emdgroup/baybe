"""Base class for surrogate-based recommenders."""

import gc
from abc import ABC, abstractmethod

import pandas as pd
from attrs import define, field
from typing_extensions import override

from baybe.objectives.base import Objective
from baybe.recommenders.pure.base import PureRecommender
from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.base import Surrogate, SurrogateProtocol


def _autoreplicate(surrogate: SurrogateProtocol, /) -> SurrogateProtocol:
    """Replicate single-output surrogate models and pass through everything else."""
    if isinstance(surrogate, Surrogate) and not surrogate.supports_multi_output:
        return surrogate.replicate()
    return surrogate


@define
class SurrogateRecommender(PureRecommender, ABC):
    """Abstract base class for recommenders that use a surrogate model."""

    _surrogate_model: SurrogateProtocol = field(
        alias="surrogate_model",
        factory=GaussianProcessSurrogate,
        converter=_autoreplicate,
    )
    """The surrogate model."""

    @override
    @abstractmethod
    def recommend(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`."""
        return super().recommend(
            batch_size,
            searchspace,
            objective,
            measurements,
            pending_experiments,
        )

    def get_surrogate(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> SurrogateProtocol:
        """Get the trained surrogate model."""
        # This fit applies internal caching and does not necessarily involve computation
        self._surrogate_model.fit(searchspace, objective, measurements)
        return self._surrogate_model


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
