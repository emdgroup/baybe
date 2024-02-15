"""Base class for all nonpredictive recommenders."""

from abc import ABC
from typing import Optional

import pandas as pd
from attrs import define

from baybe.recommenders.base import Recommender, _select_candidates_and_recommend
from baybe.searchspace import (
    SearchSpace,
    SearchSpaceType,
    SubspaceContinuous,
    SubspaceDiscrete,
)


@define
class NonPredictiveRecommender(Recommender, ABC):
    """Abstract base class for recommenders that are non-predictive."""

    def recommend(  # noqa: D102
        self,
        searchspace: SearchSpace,
        batch_size: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # See base class.

        if searchspace.type == SearchSpaceType.DISCRETE:
            return _select_candidates_and_recommend(
                searchspace,
                self._recommend_discrete,
                batch_size,
                self.allow_repeated_recommendations,
                self.allow_recommending_already_measured,
            )
        if searchspace.type == SearchSpaceType.CONTINUOUS:
            return self._recommend_continuous(
                subspace_continuous=searchspace.continuous, batch_size=batch_size
            )
        return self._recommend_hybrid(searchspace=searchspace, batch_size=batch_size)

    def _recommend_discrete(
        self,
        subspace_discrete: SubspaceDiscrete,
        candidates_comp: pd.DataFrame,
        batch_size: int,
    ) -> pd.Index:
        """Calculate recommendations in a discrete search space.

        Args:
            subspace_discrete: The discrete subspace in which the recommendations
                should be made.
            candidates_comp: The computational representation of all possible candidates
            batch_size: The size of the calculated batch.

        Raises:
            NotImplementedError: If the function is not implemented by the child class.

        Returns:
            The indices of the recommended points with respect to the
            computational representation.
        """
        try:
            return self._recommend_hybrid(
                searchspace=SearchSpace(
                    discrete=subspace_discrete, continuous=SubspaceContinuous.empty()
                ),
                batch_size=batch_size,
                candidates_comp=candidates_comp,
            ).index
        except NotImplementedError as exc:
            raise NotImplementedError(
                """Hybrid recommender could not be used as fallback when trying to
                optimize a discrete space. This is probably due to your search space and
                recommender not being compatible. Please verify that your search space
                is purely discrete and that you are either using a discrete or hybrid
                recommender."""
            ) from exc

    def _recommend_continuous(
        self, subspace_continuous: SubspaceContinuous, batch_size: int
    ) -> pd.DataFrame:
        """Calculate recommendations in a continuous search space.

        Args:
            subspace_continuous: The continuous subspace in which the recommendations
                should be made.
            batch_size: The size of the calculated batch.

        Raises:
            NotImplementedError: If the function is not implemented by the child class.

        Returns:
            The recommended points.
        """
        # If this method is not implemented by a children class, try to call
        # _recommend_hybrid instead.
        try:
            return self._recommend_hybrid(
                searchspace=SearchSpace(
                    discrete=SubspaceDiscrete.empty(), continuous=subspace_continuous
                ),
                batch_size=batch_size,
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                """Hybrid recommender could not be used as fallback when trying to
                optimize a continuous space. This is probably due to your search space
                and  recommender not being compatible. Please verify that your
                search space is purely continuous and that you are either using a
                continuous or hybrid recommender."""
            ) from exc

    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        batch_size: int,
        candidates_comp: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Calculate recommendations in a hybrid search space.

        If the recommender does not implement additional functions for discrete and
        continuous search spaces, this method is used as a fallback for those spaces
        as well.

        Args:
            searchspace: The hybrid search space in which the recommendations should
                be made.
            batch_size: The size of the calculated batch.
            candidates_comp: The computational representation of the candidates. This
                is necessary for using this function as a fallback mechanism for
                recommendations in discrete search spaces.

        Raises:
            NotImplementedError: If the function is not implemented by the child class.

        Returns:
            The recommended points.
        """
        raise NotImplementedError("Hybrid recommender is not implemented.")
