"""Base classes for all meta recommenders."""

from abc import ABC, abstractmethod
from typing import Any

import cattrs
import pandas as pd
from attrs import define

from baybe.objectives.base import Objective
from baybe.recommenders.base import RecommenderProtocol
from baybe.recommenders.deprecation import structure_recommender_protocol
from baybe.recommenders.pure.base import PureRecommender
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpace
from baybe.serialization import SerialMixin, converter, unstructure_base


@define
class MetaRecommender(SerialMixin, RecommenderProtocol, ABC):
    """Abstract base class for all meta recommenders."""

    @abstractmethod
    def select_recommender(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
    ) -> PureRecommender:
        """Select a pure recommender for the given experimentation context.

        Args:
            batch_size:
                See :func:`baybe.recommenders.meta.base.MetaRecommender.recommend`.
            searchspace:
                See :func:`baybe.recommenders.meta.base.MetaRecommender.recommend`.
            objective:
                See :func:`baybe.recommenders.meta.base.MetaRecommender.recommend`.
            measurements:
                See :func:`baybe.recommenders.meta.base.MetaRecommender.recommend`.

        Returns:
            The selected recommender.
        """

    def recommend(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """See :func:`baybe.recommenders.base.RecommenderProtocol.recommend`."""
        recommender = self.select_recommender(
            batch_size=batch_size,
            searchspace=searchspace,
            objective=objective,
            measurements=measurements,
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
            batch_size=batch_size, searchspace=searchspace, **optional_args
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
converter.register_structure_hook(MetaRecommender, structure_recommender_protocol)
