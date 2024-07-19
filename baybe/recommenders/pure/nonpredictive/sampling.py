"""Recommenders based on sampling."""

import warnings
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from baybe.exceptions import UnusedObjectWarning
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpace, SearchSpaceType, SubspaceDiscrete
from baybe.utils.sampling_algorithms import farthest_point_sampling


class RandomRecommender(NonPredictiveRecommender):
    """Recommends experiments randomly."""

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID
    # See base class.

    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        candidates_comp: pd.DataFrame,
        batch_size: int,
        pending_comp: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # See base class.

        if (pending_comp is not None) and (len(pending_comp) != 0):
            warnings.warn(
                f"'{self.recommend.__name__}' was called with a non-empty "
                f"set of pending measurements but '{self.__class__.__name__}' does not "
                f"utilize this information, meaning that the argument is ignored.",
                UnusedObjectWarning,
            )

        if searchspace.type == SearchSpaceType.DISCRETE:
            return candidates_comp.sample(batch_size)

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


class FPSRecommender(NonPredictiveRecommender):
    """An initial recommender that selects candidates via Farthest Point Sampling."""

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.DISCRETE
    # See base class.

    def _recommend_discrete(
        self,
        subspace_discrete: SubspaceDiscrete,
        candidates_comp: pd.DataFrame,
        batch_size: int,
        pending_comp: pd.DataFrame | None = None,
    ) -> pd.Index:
        # See base class.

        # Fit scaler on entire search space
        # TODO [Scaling]: scaling should be handled by search space object
        scaler = StandardScaler()
        scaler.fit(subspace_discrete.comp_rep)

        # Ignore exact pending point matches in the candidates
        if pending_comp is not None:
            candidates_comp = (
                candidates_comp.merge(pending_comp, indicator=True, how="outer")
                .query('_merge == "left_only"')
                .drop(columns=["_merge"])
            )

        # Scale and sample
        candidates_scaled = np.ascontiguousarray(scaler.transform(candidates_comp))
        ilocs = farthest_point_sampling(candidates_scaled, batch_size)
        return candidates_comp.index[ilocs]
