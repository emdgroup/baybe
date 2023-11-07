"""Strategies that switch recommenders depending on the experimentation progress."""

from typing import Iterable, Optional

import pandas as pd
from attr import define, field

from baybe.recommenders import RandomRecommender, SequentialGreedyRecommender
from baybe.recommenders.base import Recommender
from baybe.searchspace import SearchSpace
from baybe.strategies.base import Strategy


@define(kw_only=True)
class SplitStrategy(Strategy):
    """A two-phased strategy that switches the recommender after some experiments.

    Args:
        initial_recommender: The initial recommender used by the strategy.
        recommender: The recommender used by the strategy after the switch.
        switch_after: The number of experiments after which the recommender is switched.
    """

    initial_recommender: Recommender = field(factory=RandomRecommender)
    recommender: Recommender = field(factory=SequentialGreedyRecommender)
    switch_after: int = field(default=1)

    def select_recommender(  # noqa: D102
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> Recommender:
        # See base class.

        return (
            self.recommender
            if len(train_x) >= self.switch_after
            else self.initial_recommender
        )


@define(kw_only=True)
class SequentialStrategy(Strategy):
    """A strategy that uses a pre-defined sequence of recommenders.

    Args:
        recommenders: An iterable providing the recommenders to be used.
    """

    recommenders: Iterable[Recommender] = field()

    def select_recommender(  # noqa: D102
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> Recommender:
        # See base class.

        return next(self.recommenders)
