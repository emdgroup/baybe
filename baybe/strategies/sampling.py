"""Recommendation strategies based on sampling."""

from typing import ClassVar, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from baybe.searchspace import SearchSpace, SearchSpaceType
from baybe.strategies.recommender import NonPredictiveRecommender
from baybe.utils.sampling_algorithms import farthest_point_sampling


class RandomRecommender(NonPredictiveRecommender):
    """
    Recommends experiments randomly.
    """

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID

    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        batch_quantity: int,
        candidates_comp: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if searchspace.type == SearchSpaceType.DISCRETE:
            if candidates_comp is None:
                raise TypeError(
                    """You did not provide a dataframe of candidates when applying the
                    random recommender to a purely discrete space. Please ensure that
                    this dataframe is not None."""
                )
            return candidates_comp.sample(batch_quantity)
        cont_random = searchspace.continuous.samples_random(n_points=batch_quantity)
        if searchspace.type == SearchSpaceType.CONTINUOUS:
            return cont_random
        disc_candidates, _ = searchspace.discrete.get_candidates(True, True)
        disc_random = disc_candidates.sample(n=batch_quantity)

        cont_random.reset_index(drop=True)
        cont_random.index = disc_random.index
        return pd.concat([disc_random, cont_random], axis=1)


class FPSRecommender(NonPredictiveRecommender):
    """An initial strategy that selects the candidates via Farthest Point Sampling."""

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.DISCRETE

    def _recommend_discrete(
        self,
        searchspace: SearchSpace,
        candidates_comp: pd.DataFrame,
        batch_quantity: int,
    ) -> pd.Index:
        """See base class."""
        # Fit scaler on entire searchspace
        # TODO [Scaling]: scaling should be handled by searchspace object
        scaler = StandardScaler()
        scaler.fit(searchspace.discrete.comp_rep)
        candidates_scaled = np.ascontiguousarray(scaler.transform(candidates_comp))
        ilocs = farthest_point_sampling(candidates_scaled, batch_quantity)
        return candidates_comp.index[ilocs]
