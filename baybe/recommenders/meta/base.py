"""Base classes for all meta recommenders."""

import gc
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from attrs import define
from typing_extensions import override

from baybe.exceptions import DeprecationError
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

    @property
    @abstractmethod
    def is_stateful(self) -> bool:
        """Boolean indicating if the meta recommender is stateful."""

    @abstractmethod
    def select_recommender(
        self,
        batch_size: int | None = None,
        searchspace: SearchSpace | None = None,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> RecommenderProtocol:
        """Select a recommender for the given experimentation context.

        See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend` for details
        on the method arguments.
        """

    def get_non_meta_recommender(
        self,
        batch_size: int | None = None,
        searchspace: SearchSpace | None = None,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> RecommenderProtocol:
        """Follow the meta recommender chain to the selected non-meta recommender.

        Recursively calls :meth:`MetaRecommender.select_recommender` until a
        non-meta recommender is encountered, which is then returned.
        Effectively, this extracts the recommender responsible for generating
        the recommendations for the specified context.

        See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend` for details
        on the method arguments.
        """
        recommender: MetaRecommender | RecommenderProtocol = self
        while isinstance(recommender, MetaRecommender):
            recommender = recommender.select_recommender(
                batch_size, searchspace, objective, measurements, pending_experiments
            )
        return recommender

    def get_current_recommender(self) -> PureRecommender:
        """Deprecated! Use :meth:`select_recommender` or
        :meth:`get_non_meta_recommender` instead.
        """  # noqa
        raise DeprecationError(
            f"'{MetaRecommender.__name__}.get_current_recommender' has been deprecated."
            f"Use '{MetaRecommender.__name__}.{self.select_recommender.__name__}' or "
            f"'{MetaRecommender.__name__}.{self.get_non_meta_recommender.__name__}' "
            f"instead."
        )

    def get_next_recommender(self) -> PureRecommender:
        """Deprecated! Use :meth:`select_recommender` or
        :meth:`get_non_meta_recommender` instead.
        """  # noqa
        raise DeprecationError(
            f"'{MetaRecommender.__name__}.get_current_recommender' has been deprecated."
            f"Use '{MetaRecommender.__name__}.{self.select_recommender.__name__}' or "
            f"'{MetaRecommender.__name__}.{self.get_non_meta_recommender.__name__}' "
            f"instead."
        )

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
converter.register_unstructure_hook(MetaRecommender, unstructure_base)
converter.register_structure_hook(
    MetaRecommender, get_base_structure_hook(MetaRecommender)
)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
