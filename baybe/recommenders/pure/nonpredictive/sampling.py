"""Recommenders based on sampling."""

from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpace, SearchSpaceType, SubspaceDiscrete
from baybe.utils.plotting import to_string
from baybe.utils.sampling_algorithms import farthest_point_sampling


class RandomRecommender(NonPredictiveRecommender):
    """Recommends experiments randomly."""

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID
    # See base class.

    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.DataFrame:
        # See base class.

        if searchspace.type == SearchSpaceType.DISCRETE:
            return candidates_exp.sample(batch_size)

        cont_random = searchspace.continuous.sample_uniform(batch_size=batch_size)
        if searchspace.type == SearchSpaceType.CONTINUOUS:
            return cont_random

        disc_candidates, _ = searchspace.discrete.get_candidates(True, True)

        # TODO decide mechanism if number of possible discrete candidates is smaller
        #  than batch size
        disc_random = disc_candidates.sample(
            n=batch_size,
            replace=len(disc_candidates) < batch_size,
        )

        cont_random.index = disc_random.index
        return pd.concat([disc_random, cont_random], axis=1)

    def __str__(self) -> str:
        fields = [to_string("Compatibility", self.compatibility, single_line=True)]
        return to_string(self.__class__.__name__, *fields)


class FPSRecommender(NonPredictiveRecommender):
    """An initial recommender that selects candidates via Farthest Point Sampling."""

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.DISCRETE
    # See base class.

    def _recommend_discrete(
        self,
        subspace_discrete: SubspaceDiscrete,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.Index:
        # See base class.

        # Fit scaler on entire search space
        # TODO [Scaling]: scaling should be handled by search space object
        scaler = StandardScaler()
        scaler.fit(subspace_discrete.comp_rep)

        # Scale and sample
        candidates_comp = subspace_discrete.transform(candidates_exp)
        candidates_scaled = np.ascontiguousarray(scaler.transform(candidates_comp))
        ilocs = farthest_point_sampling(candidates_scaled, batch_size)
        return candidates_comp.index[ilocs]

    def __str__(self) -> str:
        fields = [to_string("Compatibility", self.compatibility, single_line=True)]
        return to_string(self.__class__.__name__, *fields)
