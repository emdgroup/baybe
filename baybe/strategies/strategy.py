"""Strategies for Design of Experiments (DOE)."""

from typing import Optional

import pandas as pd
from attrs import define, field

from baybe.recommenders.bayesian import SequentialGreedyRecommender
from baybe.recommenders.recommender import Recommender
from baybe.recommenders.sampling import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.utils import SerialMixin


@define
class Strategy(SerialMixin):
    """Abstract base class for all DOE strategies.

    Args:
        initial_recommender: The initial recommender used by the strategy.
        recommender: The recommender used by the strategy.
        allow_repeated_recommendations: Allow to make recommendations that were
            already recommended earlier. This only has an influence in discrete
            search spaces.
        allow_recommending_already_measured: Allow to output recommendations that
            were measured previously. This only has an influence in discrete
            search spaces.
    """

    initial_recommender: Recommender = field(factory=RandomRecommender)
    recommender: Recommender = field(factory=SequentialGreedyRecommender)
    allow_repeated_recommendations: bool = field(default=False)
    allow_recommending_already_measured: bool = field(default=False)

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
        recommender = (
            self.initial_recommender if len(train_x) == 0 else self.recommender
        )
        rec = recommender.recommend(
            searchspace,
            batch_quantity,
            train_x,
            train_y,
            self.allow_repeated_recommendations,
            self.allow_recommending_already_measured,
        )

        return rec
