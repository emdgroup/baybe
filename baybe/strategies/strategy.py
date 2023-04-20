# pylint: disable=missing-function-docstring

"""
Strategies for Design of Experiments (DOE).
"""

from typing import Optional

import cattrs
import pandas as pd
from attrs import define, Factory

from baybe.searchspace import SearchSpace
from baybe.strategies.bayesian import GreedyRecommender
from baybe.strategies.recommender import Recommender
from baybe.strategies.sampling import RandomRecommender


@define
class Strategy:
    """Abstract base class for all DOE strategies."""

    initial_recommender: Recommender = Factory(RandomRecommender)
    recommender: Recommender = Factory(GreedyRecommender)
    allow_repeated_recommendations: bool = True
    allow_recommending_already_measured: bool = True

    def recommend(
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
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
            batch_quantity,
            train_x,
            train_y,
            self.allow_repeated_recommendations,
            self.allow_recommending_already_measured,
        )

        return rec

    def to_dict(self):
        return cattrs.unstructure(self)

    @classmethod
    def from_dict(cls, dictionary) -> "Strategy":
        return cattrs.structure(dictionary, cls)
