"""Base classes for all strategies."""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from attrs import define, field

from baybe.recommenders.base import Recommender
from baybe.searchspace import SearchSpace
from baybe.strategies.deprecation import structure_strategy
from baybe.utils.serialization import SerialMixin, converter, unstructure_base


@define
class Strategy(SerialMixin, ABC):
    """Abstract base class for all BayBE strategies."""

    allow_repeated_recommendations: bool = field(default=False, kw_only=True)
    """Allow to make recommendations that were already recommended earlier. This only
    has an influence in discrete search spaces."""

    allow_recommending_already_measured: bool = field(default=False, kw_only=True)
    """Allow to output recommendations that were measured previously. This only has an
    influence in discrete search spaces."""

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
        """Recommend the next experiments to be conducted.

        Args:
            searchspace: The search space in which the experiments are conducted.
            batch_quantity: The number of experiments to be conducted in parallel.
            train_x: The features of the conducted experiments.
            train_y: The corresponding response values.

        Returns:
            The DataFrame with the specific experiments recommended.
        """
        recommender = self.select_recommender(
            searchspace,
            batch_quantity,
            train_x,
            train_y,
        )
        return recommender.recommend(
            searchspace,
            batch_quantity,
            train_x,
            train_y,
            self.allow_repeated_recommendations,
            self.allow_recommending_already_measured,
        )


# Register (un-)structure hooks
converter.register_unstructure_hook(Strategy, unstructure_base)
converter.register_structure_hook(Strategy, structure_strategy)
