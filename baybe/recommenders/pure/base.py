"""Base classes for all pure recommenders."""

import gc
from abc import ABC
from typing import ClassVar

import pandas as pd
from attrs import define, field

from baybe.exceptions import NotEnoughPointsLeftError
from baybe.objectives.base import Objective
from baybe.recommenders.base import RecommenderProtocol
from baybe.searchspace import SearchSpace
from baybe.searchspace.continuous import SubspaceContinuous
from baybe.searchspace.core import SearchSpaceType
from baybe.searchspace.discrete import SubspaceDiscrete


# TODO: Slots are currently disabled since they also block the monkeypatching necessary
#   to use `register_hooks`. Probably, we need to update our documentation and
#   explain how to work around that before we re-enable slots.
@define(slots=False)
class PureRecommender(ABC, RecommenderProtocol):
    """Abstract base class for all pure recommenders."""

    # Class variables
    compatibility: ClassVar[SearchSpaceType]
    """Class variable reflecting the search space compatibility."""

    # Object variables
    allow_repeated_recommendations: bool = field(default=False, kw_only=True)
    """Allow to make recommendations that were already recommended earlier.
    This only has an influence in discrete search spaces."""

    allow_recommending_already_measured: bool = field(default=True, kw_only=True)
    """Allow to make recommendations that were measured previously.
    This only has an influence in discrete search spaces."""

    allow_recommending_pending_experiments: bool = field(default=False, kw_only=True)
    """Allow `pending_experiments` to be part of the recommendations. If set to `False`,
    the corresponding points will be removed from the candidates. This only has an
    influence in discrete search spaces."""

    def recommend(  # noqa: D102
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # See base class
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
        # Repeated recommendations are always allowed for hybrid spaces
        # Pending experiments are excluded for discrete spaces unless configured
        # differently.
        dont_exclude_pending = (
            is_hybrid_space or self.allow_recommending_pending_experiments
        )
        candidates_exp, _ = searchspace.discrete.get_candidates(
            allow_repeated_recommendations=is_hybrid_space
            or self.allow_repeated_recommendations,
            allow_recommending_already_measured=is_hybrid_space
            or self.allow_recommending_already_measured,
            exclude=None if dont_exclude_pending else pending_experiments,
        )

        # TODO: Introduce new flag to recommend batches larger than the search space

        # Check if enough candidates are left
        # TODO [15917]: This check is not perfectly correct.
        if (not is_hybrid_space) and (len(candidates_exp) < batch_size):
            raise NotEnoughPointsLeftError(
                f"Using the current settings, there are fewer than {batch_size} "
                "possible data points left to recommend. This can be "
                "either because all data points have been measured at some point "
                "(while 'allow_repeated_recommendations' or "
                "'allow_recommending_already_measured' being False) "
                "or because all data points are marked as 'dont_recommend'."
            )

        # Get recommendations
        if is_hybrid_space:
            rec = self._recommend_hybrid(searchspace, candidates_exp, batch_size)
            idxs = rec.index
        else:
            idxs = self._recommend_discrete(
                searchspace.discrete, candidates_exp, batch_size
            )
            rec = searchspace.discrete.exp_rep.loc[idxs, :]

        # Update metadata
        searchspace.discrete.metadata.loc[idxs, "was_recommended"] = True

        # Return recommendations
        return rec


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
