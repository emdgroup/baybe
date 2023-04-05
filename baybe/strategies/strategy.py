"""
Strategies for Design of Experiments (DOE).
"""

from __future__ import annotations

import pandas as pd
from pydantic import Extra, Field

from baybe.searchspace import SearchSpace
from baybe.strategies.bayesian import GreedyRecommender
from baybe.strategies.recommender import Recommender
from baybe.strategies.sampling import RandomRecommender
from baybe.utils import BaseModel


class Strategy(BaseModel, extra=Extra.forbid, arbitrary_types_allowed=True):
    """Abstract base class for all DOE strategies."""

    initial_recommender: Recommender = Field(default_factory=RandomRecommender)
    recommender: Recommender = Field(default_factory=GreedyRecommender)
    allow_repeated_recommendations: bool = True
    allow_recommending_already_measured: bool = True

    def recommend(
        self,
        searchspace: SearchSpace,
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
        batch_quantity: int = 1,
    ) -> pd.DataFrame:
        """
        Recommends the next experiments to be conducted.

        Parameters
        ----------
        train_x : pd.DataFrame
            The features of the conducted experiments.
        train_y : pd.DataFrame
            The corresponding response values.
        batch_quantity : int (default = 1)
            The number of experiments to be conducted in parallel.

        Returns
        -------
        The DataFrame with the specific experiments recommended.
        """
        recommender = (
            self.initial_recommender if len(train_x) == 0 else self.recommender
        )
        rec = recommender.recommend(
            searchspace,
            train_x,
            train_y,
            batch_quantity,
            self.allow_repeated_recommendations,
            self.allow_recommending_already_measured,
        )

        return rec
