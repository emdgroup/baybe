"""Base classes for all pure recommenders."""

import gc
from abc import ABC
from typing import ClassVar, NoReturn

import pandas as pd
from attrs import define, field
from typing_extensions import override

from baybe.exceptions import DeprecationError, NotEnoughPointsLeftError
from baybe.objectives.base import Objective
from baybe.recommenders.base import RecommenderProtocol
from baybe.searchspace import SearchSpace
from baybe.searchspace.continuous import SubspaceContinuous
from baybe.searchspace.core import SearchSpaceType
from baybe.searchspace.discrete import SubspaceDiscrete

_DEPRECATION_ERROR_MESSAGE = (
    "The attribute '{}' is no longer available for recommenders. "
    "All 'allow_*' flags are now handled by `baybe.campaign.Campaign`."
)


# TODO: Slots are currently disabled since they also block the monkeypatching necessary
#   to use `register_hooks`. Probably, we need to update our documentation and
#   explain how to work around that before we re-enable slots.
@define(slots=False)
class PureRecommender(ABC, RecommenderProtocol):
    """Abstract base class for all pure recommenders."""

    compatibility: ClassVar[SearchSpaceType]
    """Class variable reflecting the search space compatibility."""

    _deprecated_allow_repeated_recommendations: bool = field(
        alias="allow_repeated_recommendations",
        default=None,
        kw_only=True,
    )
    "Deprecated! Now handled by :class:`baybe.campaign.Campaign`."

    _deprecated_allow_recommending_already_measured: bool = field(
        alias="allow_recommending_already_measured",
        default=None,
        kw_only=True,
    )
    "Deprecated! Now handled by :class:`baybe.campaign.Campaign`."

    _deprecated_allow_recommending_pending_experiments: bool = field(
        alias="allow_recommending_pending_experiments",
        default=None,
        kw_only=True,
    )
    "Deprecated! Now handled by :class:`baybe.campaign.Campaign`."

    def __attrs_post_init__(self):
        if (
            self._deprecated_allow_repeated_recommendations is not None
            or self._deprecated_allow_recommending_already_measured is not None
            or self._deprecated_allow_recommending_pending_experiments is not None
        ):
            raise DeprecationError(
                "Passing 'allow_*' flags to recommenders is no longer supported. "
                "These are now handled by `baybe.campaign.Campaign`. "
                "(Note: 'allow_repeated_recommendations' has been renamed to "
                "'allow_recommending_already_recommended'.)"
            )

    @property
    def allow_repeated_recommendations(self) -> NoReturn:
        """Deprecated!"""
        raise DeprecationError(
            _DEPRECATION_ERROR_MESSAGE.format("allow_repeated_recommendations")
        )

    @property
    def allow_recommending_already_measured(self) -> NoReturn:
        """Deprecated!"""
        raise DeprecationError(
            _DEPRECATION_ERROR_MESSAGE.format("allow_recommending_already_measured")
        )

    @property
    def allow_recommending_pending_experiments(self) -> NoReturn:
        """Deprecated!"""
        raise DeprecationError(
            _DEPRECATION_ERROR_MESSAGE.format("allow_recommending_pending_experiments")
        )

    @override
    def recommend(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if searchspace.type is SearchSpaceType.CONTINUOUS:
            return self._recommend_continuous(
                subspace_continuous=searchspace.continuous, batch_size=batch_size
            )
        else:
            return self._recommend_with_discrete_parts(
                searchspace,
                batch_size,
                pending_experiments=pending_experiments,
            )

    def _recommend_discrete(
        self,
        subspace_discrete: SubspaceDiscrete,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.Index:
        """Generate recommendations from a discrete search space.

        Args:
            subspace_discrete: The discrete subspace from which to generate
                recommendations.
            candidates_exp: The experimental representation of all discrete candidate
                points to be considered.
            batch_size: The size of the recommendation batch.

        Raises:
            NotImplementedError: If the function is not implemented by the child class.

        Returns:
            The dataframe indices of the recommended points in the provided
            experimental representation.
        """
        # If this method is not implemented by a child class, try to resort to hybrid
        # recommendation (with an empty subspace) instead.
        try:
            return self._recommend_hybrid(
                searchspace=SearchSpace(discrete=subspace_discrete),
                candidates_exp=candidates_exp,
                batch_size=batch_size,
            ).index
        except NotImplementedError as exc:
            raise NotImplementedError(
                """Hybrid recommendation could not be used as fallback when trying to
                optimize a discrete space.
                This is probably due to your search space and recommender not being
                compatible.
                If you operate in discrete search spaces,
                ensure that you either use a discrete or a hybrid recommender."""
            ) from exc

    def _recommend_continuous(
        self,
        subspace_continuous: SubspaceContinuous,
        batch_size: int,
    ) -> pd.DataFrame:
        """Generate recommendations from a continuous search space.

        Args:
            subspace_continuous: The continuous subspace from which to generate
                recommendations.
            batch_size: The size of the recommendation batch.

        Raises:
            NotImplementedError: If the function is not implemented by the child class.

        Returns:
            A dataframe containing the recommendations as individual rows.
        """
        # If this method is not implemented by a child class, try to resort to hybrid
        # recommendation (with an empty subspace) instead.
        try:
            return self._recommend_hybrid(
                searchspace=SearchSpace(continuous=subspace_continuous),
                candidates_exp=pd.DataFrame(),
                batch_size=batch_size,
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                """Hybrid recommendation could not be used as fallback when trying to
                optimize a continuous space.
                This is probably due to your search space and recommender not being
                compatible.
                If you operate in continuous search spaces,
                ensure that you either use a continuous or a hybrid recommender."""
            ) from exc

    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.DataFrame:
        """Generate recommendations from a hybrid search space.

        If the recommender does not implement additional functions for discrete and
        continuous search spaces, this method is used as a fallback for those spaces
        as well.

        Args:
            searchspace: The hybrid search space from which to generate
                recommendations.
            candidates_exp: The experimental representation of all discrete candidate
                points to be considered.
            batch_size: The size of the recommendation batch.

        Raises:
            NotImplementedError: If the function is not implemented by the child class.

        Returns:
            A dataframe containing the recommendations as individual rows.
        """
        raise NotImplementedError("Hybrid recommendation is not implemented.")

    def _recommend_with_discrete_parts(
        self,
        searchspace: SearchSpace,
        batch_size: int,
        pending_experiments: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Obtain recommendations in search spaces with a discrete part.

        Convenience helper which sequentially performs the following tasks: get discrete
        candidates, generate recommendations, update metadata.

        Args:
            searchspace: The search space from which to generate recommendations.
            batch_size: The size of the recommendation batch.
            pending_experiments: Pending experiments in experimental representation.

        Returns:
            A dataframe containing the recommendations as individual rows.

        Raises:
            NotEnoughPointsLeftError: If there are fewer points left for potential
                recommendation than requested.
        """
        is_hybrid_space = searchspace.type is SearchSpaceType.HYBRID

        # Get discrete candidates
        candidates_exp, _ = searchspace.discrete.get_candidates()

        # TODO: Introduce new flag to recommend batches larger than the search space

        # Check if enough candidates are left
        # TODO [15917]: This check is not perfectly correct.
        if (not is_hybrid_space) and (len(candidates_exp) < batch_size):
            raise NotEnoughPointsLeftError(
                f"Using the current settings, there are fewer than {batch_size} "
                f"possible data points left to recommend."
            )

        # Get recommendations
        if is_hybrid_space:
            rec = self._recommend_hybrid(searchspace, candidates_exp, batch_size)
        else:
            idxs = self._recommend_discrete(
                searchspace.discrete, candidates_exp, batch_size
            )
            rec = searchspace.discrete.exp_rep.loc[idxs, :]

        # Return recommendations
        return rec


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
