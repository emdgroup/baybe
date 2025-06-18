"""Recommenders based on sampling."""

import warnings
from enum import Enum
from typing import ClassVar

import numpy as np
import pandas as pd
from attrs import define, field
from attrs.validators import instance_of
from sklearn.preprocessing import StandardScaler
from typing_extensions import override

from baybe.exceptions import OptionalImportError
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpace, SearchSpaceType
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


class FPSInitialization(Enum):
    """Initialization methods for farthest point sampling."""

    FARTHEST = "farthest"
    """Selects the first two points with the largest distance."""

    RANDOM = "random"
    """Selects the first point uniformly at random."""


@define
class FPSRecommender(NonPredictiveRecommender):
    """An initial recommender that selects candidates via Farthest Point Sampling."""

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID
    # See base class.

    initialization: FPSInitialization = field(
        default=FPSInitialization.FARTHEST,
        converter=FPSInitialization,
    )
    """See :func:`baybe.utils.sampling_algorithms.farthest_point_sampling`."""

    random_tie_break: bool = field(
        default=True, validator=instance_of(bool), kw_only=True
    )
    """See :func:`baybe.utils.sampling_algorithms.farthest_point_sampling`."""

    grid_density: int = field(default=10, kw_only=True)
    """Number of points to sample per dimension for continuous subspaces."""

    @override
    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.DataFrame:
        # If space is purely discrete, reuse _recommend_discrete logic
        if searchspace.type == SearchSpaceType.DISCRETE:
            return candidates_exp.sample(
                n=batch_size,
                replace=len(candidates_exp) < batch_size,
            )

        # Sample a grid from continuous subspace
        grid_density = self.grid_density

        cont_bounds = searchspace.continuous.comp_rep_bounds
        param_names = searchspace.continuous.parameter_names

        grid_axes = [
            np.linspace(start, stop, num=grid_density)
            for start, stop in zip(cont_bounds.iloc[0], cont_bounds.iloc[1])
        ]

        grid = np.meshgrid(*grid_axes, indexing="ij")
        grid_points = np.stack([g.ravel() for g in grid], axis=-1)
        cont_grid_df = pd.DataFrame(grid_points, columns=param_names)

        # For hybrid, combine discrete and continuous points
        if searchspace.type == SearchSpaceType.HYBRID:
            disc_candidates, _ = searchspace.discrete.get_candidates()

            total_grid = pd.concat(
                [
                    disc_candidates.loc[
                        disc_candidates.index.repeat(len(cont_grid_df))
                    ].reset_index(drop=True),
                    pd.concat([cont_grid_df] * len(disc_candidates), ignore_index=True),
                ],
                axis=1,
            )
        else:
            total_grid = cont_grid_df

        # Transform, scale and then sample
        scaler = StandardScaler()
        candidates_comp = searchspace.transform(total_grid)
        candidates_scaled = scaler.fit_transform(candidates_comp)

        # Try fpsample first
        try:
            from baybe._optional import fpsample

            if self.initialization != FPSInitialization.FARTHEST:
                warnings.warn(
                    f"{self.__class__.__name__} is using the optional 'fpsample', "
                    f"which does not support '{self.initialization.value}'. "
                    f"Please uninstall 'fpsample' or choose a supported initialization."
                )
            if self.random_tie_break:
                warnings.warn(
                    f"{self.__class__.__name__} is using the optional 'fpsample', "
                    f"which does not support random tie-breaking. "
                    f"Selection will follow a deterministic order. "
                )
            ilocs = fpsample.fps_sampling(
                candidates_scaled,
                n_samples=batch_size,
            )
        except OptionalImportError:
            # Custom implementation as fallback
            ilocs = farthest_point_sampling(
                candidates_scaled,
                batch_size,
                initialization=self.initialization.value,
                random_tie_break=self.random_tie_break,
            )
        return total_grid.iloc[ilocs]

    @override
    def __str__(self) -> str:
        fields = [to_string("Compatibility", self.compatibility, single_line=True)]
        return to_string(self.__class__.__name__, *fields)
