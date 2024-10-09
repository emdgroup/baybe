"""Base classes for all meta recommenders."""

import gc
from abc import ABC, abstractmethod
from typing import Any

import cattrs
import pandas as pd
from attrs import define, field

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

    _current_recommender: PureRecommender | None = field(default=None, init=False)
    """The current recommender."""

    _used_recommender_ids: set[int] = field(factory=set, init=False)
    """Set of ids from recommenders that were used by this meta recommender."""

    @abstractmethod
    def select_recommender(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> PureRecommender:
        """Select a pure recommender for the given experimentation context.

        See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend` for details
        on the method arguments.
        """

    def get_current_recommender(self) -> PureRecommender:
        """Get the current recommender, if available."""
        if self._current_recommender is None:
            raise RuntimeError(
                f"No recommendation has been requested from the "
                f"'{self.__class__.__name__}' yet. Because the recommender is a "
                f"'{MetaRecommender.__name__}', this means no actual recommender has "
                f"been selected so far. The recommender will be available after the "
                f"next '{self.recommend.__name__}' call."
            )
        return self._current_recommender

    def get_next_recommender(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> PureRecommender:
        """Get the recommender for the next recommendation.

        Returns the next recommender in row that has not yet been used for generating
        recommendations. In case of multiple consecutive calls, this means that
        the same recommender instance is returned until its :meth:`recommend` method
        is called.

        See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend` for details
        on the method arguments.
        """
        # Check if the stored recommender instance can be returned
        if (
            self._current_recommender is not None
            and id(self._current_recommender) not in self._used_recommender_ids
        ):
            recommender = self._current_recommender

        # Otherwise, fetch the next recommender waiting in row
        else:
            recommender = self.select_recommender(
                batch_size=batch_size,
                searchspace=searchspace,
                objective=objective,
                measurements=measurements,
                pending_experiments=pending_experiments,
            )
            self._current_recommender = recommender

        return recommender

    def recommend(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`."""
        recommender = self.get_next_recommender(
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

        recommendations = recommender.recommend(
            batch_size=batch_size,
            searchspace=searchspace,
            pending_experiments=pending_experiments,
            **optional_args,
        )
        self._used_recommender_ids.add(id(recommender))

        return recommendations


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
