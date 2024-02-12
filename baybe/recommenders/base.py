"""Base classes for all recommenders."""

from abc import ABC
from typing import ClassVar, Optional, Protocol

import cattrs
import pandas as pd
from attrs import define, field

from baybe.exceptions import NotEnoughPointsLeftError
from baybe.recommenders.deprecation import structure_recommender_protocol
from baybe.searchspace import SearchSpace, SearchSpaceType, SubspaceDiscrete
from baybe.searchspace.continuous import SubspaceContinuous
from baybe.serialization import converter, unstructure_base


class RecommenderProtocol(Protocol):
    """Type protocol specifying the interface recommenders need to implement."""

    def recommend(
        self,
        searchspace: SearchSpace,
        batch_size: int,
        train_x: Optional[pd.DataFrame],
        train_y: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Recommend a batch of points from the given search space.

        Args:
            searchspace: The search space from which to recommend the points.
            batch_size: The number of points to be recommended.
            train_x: Optional training inputs for training a model.
            train_y: Optional training labels for training a model.

        Returns:
            A dataframe containing the recommendations as individual rows.
        """
        ...


@define
class Recommender(ABC, RecommenderProtocol):
    """Abstract base class for all recommenders."""

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

    def recommend(  # noqa: D102
        self,
        searchspace: SearchSpace,
        batch_size: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # See base class.

        if searchspace.type == SearchSpaceType.DISCRETE:
            # Select, recommend, and mark discrete candidates in one atomic step
            return self._select_candidates_and_recommend(
                searchspace.discrete,
                batch_size,
            )

        if searchspace.type == SearchSpaceType.CONTINUOUS:
            return self._recommend_continuous(
                subspace_continuous=searchspace.continuous, batch_size=batch_size
            )

        if searchspace.type == SearchSpaceType.HYBRID:
            # Ignore the flags in hybrid spaces
            _, candidates_comp = searchspace.discrete.get_candidates(
                allow_repeated_recommendations=True,
                allow_recommending_already_measured=True,
            )
            return self._recommend_hybrid(
                searchspace=searchspace,
                candidates_comp=candidates_comp,
                batch_size=batch_size,
            )

        raise RuntimeError("This line should be impossible to reach.")

    def _recommend_discrete(
        self,
        subspace_discrete: SubspaceDiscrete,
        candidates_comp: pd.DataFrame,
        batch_size: int,
    ) -> pd.Index:
        """Generate recommendations from a discrete search space.

        Args:
            subspace_discrete: The discrete subspace from which to generate
                recommendations.
            candidates_comp: The computational representation of all discrete candidate
                points to be considered.
            batch_size: The size of the recommendation batch.

        Raises:
            NotImplementedError: If the function is not implemented by the child class.

        Returns:
            The dataframe indices of the recommended points in the provided
            computational representation.
        """
        # If this method is not implemented by a child class, try to resort to hybrid
        # recommendation (with an empty subspace) instead.
        try:
            return self._recommend_hybrid(
                searchspace=SearchSpace(discrete=subspace_discrete),
                candidates_comp=candidates_comp,
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
                candidates_comp=pd.DataFrame(),
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
        candidates_comp: pd.DataFrame,
        batch_size: int,
    ) -> pd.DataFrame:
        """Generate recommendations from a hybrid search space.

        If the recommender does not implement additional functions for discrete and
        continuous search spaces, this method is used as a fallback for those spaces
        as well.

        Args:
            searchspace: The hybrid search space from which to generate
                recommendations.
            candidates_comp: The computational representation of all discrete candidate
                points to be considered.
            batch_size: The size of the recommendation batch.

        Raises:
            NotImplementedError: If the function is not implemented by the child class.

        Returns:
            A dataframe containing the recommendations as individual rows.
        """
        raise NotImplementedError("Hybrid recommendation is not implemented.")

    def _select_candidates_and_recommend(
        self,
        subspace_discrete: SubspaceDiscrete,
        batch_size: int,
    ) -> pd.DataFrame:
        """Get candidates in a discrete search space and generate recommendations.

        Args:
            subspace_discrete: The discrete subspace from which to generate
                recommendations.
            batch_size: The size of the recommendation batch.

        Returns:
            A dataframe containing the recommendations as individual rows.

        Raises:
            NotEnoughPointsLeftError: If there are fewer points left for potential
                recommendation than requested.
        """
        # Get discrete candidates
        _, candidates_comp = subspace_discrete.get_candidates(
            allow_repeated_recommendations=self.allow_repeated_recommendations,
            allow_recommending_already_measured=self.allow_recommending_already_measured,
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
        idxs = self._recommend_discrete(subspace_discrete, candidates_comp, batch_size)
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
