"""Recommenders based on sampling."""

from enum import Enum
from typing import ClassVar

import numpy as np
import pandas as pd
from attrs import define, field, fields
from attrs.validators import instance_of
from typing_extensions import override

from baybe.exceptions import InfeasibilityError
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpace, SearchSpaceType, SubspaceDiscrete
from baybe.settings import Settings, active_settings
from baybe.utils.conversion import to_string
from baybe.utils.sampling_algorithms import farthest_point_sampling


class RandomRecommender(NonPredictiveRecommender):
    """Recommends experiments randomly."""

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID
    # See base class.

    supports_discrete_subset_generating_constraints: ClassVar[bool] = True
    # See base class.

    @override
    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        batch_size: int,
    ) -> pd.DataFrame:
        is_hybrid = searchspace.type is SearchSpaceType.HYBRID

        # Sample continuous part if applicable
        if is_hybrid or searchspace.type is SearchSpaceType.CONTINUOUS:
            cont_random = searchspace.continuous.sample_uniform(batch_size=batch_size)
            if searchspace.type is SearchSpaceType.CONTINUOUS:
                return cont_random

        candidates_exp = searchspace.discrete.get_candidates()

        # Restrict to a random subset if subset-generating constraints are present
        if searchspace.discrete.n_subsets > 0:
            masks = searchspace.discrete.sample_subset_masks(
                n=1,
                min_candidates=None if is_hybrid else batch_size,
            )
            if not masks:
                raise InfeasibilityError(
                    "No feasible subset found for the given "
                    "subset-generating constraints. All subsets have fewer "
                    f"candidates than the requested {batch_size=}."
                )
            candidates_exp = candidates_exp.loc[masks[0]]

        disc_random = candidates_exp.sample(
            n=batch_size,
            replace=is_hybrid or len(candidates_exp) < batch_size,
        )

        if not is_hybrid:
            return disc_random

        cont_random.index = disc_random.index
        return pd.concat([disc_random, cont_random], axis=1)

    @override
    def __str__(self) -> str:
        fields = [to_string("Compatibility", self.compatibility, single_line=True)]
        return to_string(self.__class__.__name__, *fields)


class FPSInitialization(Enum):
    """Initialization methods for farthest point sampling."""

    FARTHEST = "farthest"
    """Selects the first two points with the largest distance."""

    RANDOM = "random"
    """Selects the first point uniformly at random."""


@define
class FPSRecommender(NonPredictiveRecommender):
    """An initial recommender that selects candidates via Farthest Point Sampling.

    If the optional `fpsample <https://github.com/leonardodalinky/fpsample>`_ package is
    installed, a more efficient implementation is available that can be (de-)activated
    via the :attr:`~baybe.settings.Settings.use_fpsample` setting. Otherwise, a custom
    fallback implementation is used.
    """

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.DISCRETE
    # See base class.

    initialization: FPSInitialization = field(
        default=FPSInitialization.FARTHEST, converter=FPSInitialization
    )
    """See :func:`~baybe.utils.sampling_algorithms.farthest_point_sampling`.

    If the optional package 'fpsample' is used, only
    :attr:`~baybe.recommenders.pure.nonpredictive.sampling.FPSInitialization.FARTHEST`
    is supported.
    """

    random_tie_break: bool = field(validator=instance_of(bool), kw_only=True)
    """See :func:`~baybe.utils.sampling_algorithms.farthest_point_sampling`.

    If the optional package 'fpsample' is used, only ``False`` is supported.
    """

    @initialization.validator
    def _validate_initialization(self, _, value):
        if active_settings.use_fpsample and value is not FPSInitialization.FARTHEST:
            raise ValueError(
                f"'{self.__class__.__name__}' is currently using the optional "
                f"'fpsample' package, which does not support the "
                f"'{self.initialization}' mode. "
                f"Please choose a supported initialization mode or deactivate "
                f"`fpsample` usage via the '{fields(Settings)._use_fpsample.alias}' "
                f"option in BayBE's settings."
            )

    @random_tie_break.default
    def _default_random_tie_break(self) -> bool:
        return self.initialization is not FPSInitialization.FARTHEST

    @random_tie_break.validator
    def _validate_random_tie_break(self, _, value):
        if active_settings.use_fpsample and value:
            raise ValueError(
                f"'{self.__class__.__name__}' is currently using the optional "
                f"'fpsample' package, which does not support random tie-breaking. "
                f"Either disable the mechanism by passing "
                f"'{fields(self.__class__).random_tie_break.name}=False' or deactivate "
                f"`fpsample` usage via the '{fields(Settings)._use_fpsample.alias}' "
                f"option in BayBE's settings."
            )

    @override
    def _recommend_discrete(
        self,
        subspace_discrete: SubspaceDiscrete,
        batch_size: int,
    ) -> pd.DataFrame:
        # Fit scaler on entire search space
        from sklearn.preprocessing import StandardScaler

        # TODO [Scaling]: scaling should be handled by search space object
        candidates_comp = subspace_discrete.transform(
            subspace_discrete.get_candidates()
        )
        scaler = StandardScaler()
        scaler.fit(candidates_comp)

        # Scale and sample
        candidates_scaled = np.ascontiguousarray(scaler.transform(candidates_comp))

        if active_settings.use_fpsample:
            from baybe._optional.fpsample import fps_sampling

            ilocs = fps_sampling(
                candidates_scaled,
                n_samples=batch_size,
            )
        else:
            ilocs = farthest_point_sampling(
                candidates_scaled,
                batch_size,
                initialization=self.initialization.value,
                random_tie_break=self.random_tie_break,
            )
        return subspace_discrete.get_candidates().iloc[ilocs]

    @override
    def __str__(self) -> str:
        fields = [to_string("Compatibility", self.compatibility, single_line=True)]
        return to_string(self.__class__.__name__, *fields)
