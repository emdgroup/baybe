"""Recommenders based on sampling."""

from typing import ClassVar, Literal

import numpy as np
import pandas as pd
from attrs import define, field
from attrs.validators import instance_of
from sklearn.preprocessing import StandardScaler
from typing_extensions import override

from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpace, SearchSpaceType, SubspaceDiscrete
from baybe.utils.conversion import to_string
from baybe.utils.sampling_algorithms import farthest_point_sampling


class RandomRecommender(NonPredictiveRecommender):
    """Recommends experiments randomly."""

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID
    # See base class.

    @override
    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.DataFrame:
        if searchspace.type == SearchSpaceType.DISCRETE:
            return candidates_exp.sample(batch_size)

        cont_random = searchspace.continuous.sample_uniform(batch_size=batch_size)
        if searchspace.type == SearchSpaceType.CONTINUOUS:
            return cont_random

        disc_candidates, _ = searchspace.discrete.get_candidates()

        # TODO decide mechanism if number of possible discrete candidates is smaller
        #  than batch size
        disc_random = disc_candidates.sample(
            n=batch_size,
            replace=len(disc_candidates) < batch_size,
        )

        cont_random.index = disc_random.index
        return pd.concat([disc_random, cont_random], axis=1)

    @override
    def __str__(self) -> str:
        fields = [to_string("Compatibility", self.compatibility, single_line=True)]
        return to_string(self.__class__.__name__, *fields)


@define(kw_only=True)
class FPSRecommender(NonPredictiveRecommender):
    """An initial recommender that selects candidates via Farthest Point Sampling."""

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.DISCRETE
    # See base class.

    initialization: Literal["farthest", "random"] = field(
        default="farthest", validator=instance_of(str)
    )
    """See :func:`baybe.utils.sampling_algorithms.farthest_point_sampling`."""

    random_tie_break: bool = field(default=True, validator=instance_of(bool))
    """See :func:`baybe.utils.sampling_algorithms.farthest_point_sampling`."""

    @override
    def _recommend_discrete(
        self,
        subspace_discrete: SubspaceDiscrete,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.Index:
        # Fit scaler on entire search space
        # TODO [Scaling]: scaling should be handled by search space object
        scaler = StandardScaler()
        scaler.fit(subspace_discrete.comp_rep)

        # Scale and sample
        candidates_comp = subspace_discrete.transform(candidates_exp)
        candidates_scaled = np.ascontiguousarray(scaler.transform(candidates_comp))
        ilocs = farthest_point_sampling(
            candidates_scaled,
            batch_size,
            initialization=self.initialization,
            random_tie_break=self.random_tie_break,
        )
        return candidates_comp.index[ilocs]

    @override
    def __str__(self) -> str:
        fields = [to_string("Compatibility", self.compatibility, single_line=True)]
        return to_string(self.__class__.__name__, *fields)
