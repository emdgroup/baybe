"""Recommendation strategies based on sampling."""

import pandas as pd

from baybe.searchspace import SearchSpace, SearchSpaceType
from baybe.strategies.recommender import NonPredictiveRecommender
from baybe.utils.sampling_algorithms import farthest_point_sampling


class RandomRecommender(NonPredictiveRecommender):
    """
    Recommends experiments randomly.
    """

    type = "RANDOM"
    compatibility = SearchSpaceType.EITHER  # TODO: enable HYBRID mode

    def _recommend_continuous(
        self, searchspace: SearchSpace, batch_quantity: int
    ) -> pd.DataFrame:
        """See base class."""
        return searchspace.continuous.samples_random(n_points=batch_quantity)

    def _recommend_discrete(
        self,
        searchspace: SearchSpace,
        candidates_comp: pd.DataFrame,
        batch_quantity: int,
    ) -> pd.Index:
        """See base class."""
        return candidates_comp.sample(n=batch_quantity).index


class FPSRecommender(NonPredictiveRecommender):
    """An initial strategy that selects the candidates via Farthest Point Sampling."""

    type = "FPS"
    compatibility = SearchSpaceType.DISCRETE

    def _recommend_discrete(
        self,
        searchspace: SearchSpace,
        candidates_comp: pd.DataFrame,
        batch_quantity: int,
    ) -> pd.Index:
        """See base class."""
        ilocs = farthest_point_sampling(candidates_comp.values, batch_quantity)
        return candidates_comp.index[ilocs]
