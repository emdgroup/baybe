"""Base classes for all strategies."""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from attrs import define

from baybe.recommenders.base import Recommender, RecommenderProtocol
from baybe.recommenders.deprecation import structure_recommender_protocol
from baybe.searchspace import SearchSpace
from baybe.serialization import SerialMixin, converter, unstructure_base


@define
class Strategy(SerialMixin, RecommenderProtocol, ABC):
    """Abstract base class for all BayBE strategies."""

    @abstractmethod
    def select_recommender(
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> Recommender:
        """Select a recommender for the given experimentation context.

        Args:
            searchspace: See :func:`baybe.strategies.base.Strategy.recommend`.
            batch_quantity: See :func:`baybe.strategies.base.Strategy.recommend`.
            train_x: See :func:`baybe.strategies.base.Strategy.recommend`.
            train_y: See :func:`baybe.strategies.base.Strategy.recommend`.

        Returns:
            The selected recommender.
        """

    def recommend(
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """See :func:`baybe.recommenders.base.RecommenderProtocol.recommend`."""
        recommender = self.select_recommender(
            searchspace, batch_quantity, train_x, train_y
        )
        return recommender.recommend(searchspace, batch_quantity, train_x, train_y)


# Register (un-)structure hooks
converter.register_unstructure_hook(Strategy, unstructure_base)
converter.register_structure_hook(Strategy, structure_recommender_protocol)
