"""Strategies that switch recommenders depending on the experimentation progress."""

from typing import Optional

import pandas as pd
from attr import define, field

from baybe.recommenders import RandomRecommender, SequentialGreedyRecommender
from baybe.recommenders.base import Recommender
from baybe.searchspace import SearchSpace
from baybe.strategies.base import Strategy


@define
class SplitStrategy(Strategy):
    """A two-phased strategy that switches the recommender after some experiments.

    Args:
        initial_recommender: The initial recommender used by the strategy.
        recommender: The recommender used by the strategy after the switch.
        switch_at: The number of experiments at which the recommender is switched.
    """

    initial_recommender: Recommender = field(factory=RandomRecommender)
    recommender: Recommender = field(factory=SequentialGreedyRecommender)
    switch_at: int = field(default=1)

    def recommend(  # noqa: D102
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # See base class.

        recommender = (
            self.recommender
            if len(train_x) >= self.switch_at
            else self.initial_recommender
        )
        return recommender.recommend(
            searchspace,
            batch_quantity,
            train_x,
            train_y,
            self.allow_repeated_recommendations,
            self.allow_recommending_already_measured,
        )
