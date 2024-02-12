"""Base classes for all recommenders."""

from abc import ABC
from typing import Callable, ClassVar, Optional, Protocol

import cattrs
import pandas as pd
from attrs import define, field

from baybe.exceptions import NotEnoughPointsLeftError
from baybe.recommenders.deprecation import structure_recommender_protocol
from baybe.searchspace import SearchSpace, SearchSpaceType, SubspaceDiscrete
from baybe.searchspace.continuous import SubspaceContinuous
from baybe.serialization import (
    converter,
    unstructure_base,
)


class RecommenderProtocol(Protocol):
    """Type protocol specifying the interface recommenders need to implement."""

    def recommend(
        self,
        searchspace: SearchSpace,
        batch_size: int,
        train_x: Optional[pd.DataFrame],
        train_y: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Recommend (a batch of) points in the search space.

        Args:
            searchspace: The search space in which experiments are being conducted.
            batch_size: The number of points that should be recommended.
            train_x: The training data used to train the model.
            train_y: The training labels used to train the model.

        Returns:
            A DataFrame containing the recommendations as individual rows.
        """
        ...


@define
class Recommender(ABC, RecommenderProtocol):
    """Abstract base class for all recommenders."""

    # Class variables
    compatibility: ClassVar[SearchSpaceType]
    """Class variable describing the search space compatibility."""

    # Object variables
    allow_repeated_recommendations: bool = field(default=False, kw_only=True)
    """Allow to make recommendations that were already recommended earlier. This only
    has an influence in discrete search spaces."""

    allow_recommending_already_measured: bool = field(default=True, kw_only=True)
    """Allow to output recommendations that were measured previously. This only has an
    influence in discrete search spaces."""

    def recommend(  # noqa: D102
        self,
        searchspace: SearchSpace,
        batch_size: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # See base class.

        if searchspace.type == SearchSpaceType.DISCRETE:
            return self._select_candidates_and_recommend(
                searchspace.discrete,
                self._recommend_discrete,
                batch_size,
                allow_repeated_recommendations=self.allow_repeated_recommendations
                or not searchspace.continuous.is_empty,
                allow_recommending_already_measured=self.allow_recommending_already_measured
                or not searchspace.continuous.is_empty,
            )
        if searchspace.type == SearchSpaceType.CONTINUOUS:
            return self._recommend_continuous(
                subspace_continuous=searchspace.continuous, batch_size=batch_size
            )
        if searchspace.type == SearchSpaceType.HYBRID:
            return self._recommend_hybrid(
                searchspace=searchspace, batch_size=batch_size
            )
        raise RuntimeError("This line should be impossible to reach.")

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

    def _select_candidates_and_recommend(
        self,
        subspace_discrete: SubspaceDiscrete,
        recommend_discrete: Callable[[SubspaceDiscrete, pd.DataFrame, int], pd.Index],
        batch_size: int = 1,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        """Select candidates in a discrete search space and recommend them.

        This function is a workaround as this functionality is required for all purely
        discrete recommenders and avoids the introduction of complicate class
        hierarchies. It is also used to select candidates in the discrete part of hybrid
        search spaces, ignoring the continuous part.

        Args:
            subspace_discrete: The discrete subspace.
            recommend_discrete: The Callable representing the discrete recommendation
                function.
            batch_size: The chosen batch size.
            allow_repeated_recommendations: Allow to make recommendations that were
                already recommended earlier.
            allow_recommending_already_measured: Allow to output recommendations that
                were measured previously.

        Returns:
            The recommendation in experimental representation.

        Raises:
            NotEnoughPointsLeftError: If there are fewer than ``batch_size`` points
                left for potential recommendation.
        """
        # IMPROVE: See if the there is a more elegant way to share this functionality
        #   among all purely discrete recommenders (without introducing complicates
        #   class hierarchies).

        # Get discrete candidates. The metadata flags are ignored if the search space
        # has a continuous component.
        _, candidates_comp = subspace_discrete.get_candidates(
            allow_repeated_recommendations=allow_repeated_recommendations,
            allow_recommending_already_measured=allow_recommending_already_measured,
        )

        # Check if enough candidates are left
        # TODO [15917]: This check is not perfectly correct.
        if len(candidates_comp) < batch_size:
            raise NotEnoughPointsLeftError(
                f"Using the current settings, there are fewer than {batch_size} "
                "possible data points left to recommend. This can be "
                "either because all data points have been measured at some point "
                "(while 'allow_repeated_recommendations' or "
                "'allow_recommending_already_measured' being False) "
                "or because all data points are marked as 'dont_recommend'."
            )

        # Get recommendations
        idxs = recommend_discrete(subspace_discrete, candidates_comp, batch_size)
        rec = subspace_discrete.exp_rep.loc[idxs, :]

        # Update metadata
        subspace_discrete.metadata.loc[idxs, "was_recommended"] = True

        # Return recommendations
        return rec


# Register (un-)structure hooks
converter.register_unstructure_hook(
    RecommenderProtocol,
    lambda x: unstructure_base(
        x,
        # TODO: Remove once deprecation got expired:
        overrides=dict(
            allow_repeated_recommendations=cattrs.override(omit=True),
            allow_recommending_already_measured=cattrs.override(omit=True),
        ),
    ),
)
converter.register_structure_hook(RecommenderProtocol, structure_recommender_protocol)
