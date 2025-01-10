"""Base classes for all meta recommenders."""

import gc
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import cattrs
import pandas as pd
from attrs import define
from typing_extensions import override

from baybe.objectives.base import Objective
from baybe.recommenders.base import RecommenderProtocol
from baybe.recommenders.pure.base import PureRecommender
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpace
from baybe.serialization import SerialMixin, converter, unstructure_base
from baybe.serialization.core import get_base_structure_hook


@define
class MetaRecommender(SerialMixin, RecommenderProtocol, ABC):
    """Abstract base class for all meta recommenders."""

    is_stateful: ClassVar[bool] = False
    """Boolean flag indicating if the meta recommender is stateful."""

    @abstractmethod
    def select_recommender(
        self,
        batch_size: int | None = None,
        searchspace: SearchSpace | None = None,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> PureRecommender:
        """Select a pure recommender for the given experimentation context.

        See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend` for details
        on the method arguments.
        """

    @override
    def recommend(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`."""
        recommender = self.select_recommender(
            batch_size=batch_size,
            searchspace=searchspace,
            objective=objective,
            measurements=measurements,
            pending_experiments=pending_experiments,
        )

        # Non-predictive recommenders should not be called with an objective or
        # measurements. Using dict value type Any here due to known mypy complication:
        # https://github.com/python/mypy/issues/5382
        optional_args: dict[str, Any] = (
            {}
            if isinstance(recommender, NonPredictiveRecommender)
            else {
                "objective": objective,
                "measurements": measurements,
            }
        )

        return recommender.recommend(
            batch_size=batch_size,
            searchspace=searchspace,
            pending_experiments=pending_experiments,
            **optional_args,
        )


# Register (un-)structure hooks
converter.register_unstructure_hook(
    MetaRecommender,
    lambda x: unstructure_base(
        x,
        # TODO: Remove once deprecation got expired:
        overrides=dict(
            allow_repeated_recommendations=cattrs.override(omit=True),
            allow_recommending_already_measured=cattrs.override(omit=True),
        ),
    ),
)
converter.register_structure_hook(
    MetaRecommender, get_base_structure_hook(MetaRecommender)
)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
