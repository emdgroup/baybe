# pylint: disable=missing-class-docstring, missing-function-docstring
# TODO: add docstrings

"""Base classes for all recommenders."""

from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Optional

import cattrs
import pandas as pd
from attrs import define

from baybe.searchspace import SearchSpace, SearchSpaceType
from baybe.utils import (
    get_base_unstructure_hook,
    NotEnoughPointsLeftError,
    unstructure_base,
)


# TODO: See if the there is a more elegant way to share this functionality
#   among all purely discrete recommenders (without introducing complicates class
#   hierarchies).
def select_candidates_and_recommend(
    searchspace: SearchSpace,
    recommend: Callable[[SearchSpace, pd.DataFrame, int], pd.Index],
    batch_quantity: int = 1,
    allow_repeated_recommendations: bool = False,
    allow_recommending_already_measured: bool = True,
) -> pd.DataFrame:
    # Get discrete candidates. The metadata flags are ignored if the searchspace
    # has a continuous component.
    _, candidates_comp = searchspace.discrete.get_candidates(
        allow_repeated_recommendations=allow_repeated_recommendations
        or not searchspace.continuous.is_empty,
        allow_recommending_already_measured=allow_recommending_already_measured
        or not searchspace.continuous.is_empty,
    )

    # Check if enough candidates are left
    if len(candidates_comp) < batch_quantity:
        raise NotEnoughPointsLeftError(
            f"Using the current settings, there are fewer than {batch_quantity} "
            "possible data points left to recommend. This can be "
            "either because all data points have been measured at some point "
            "(while 'allow_repeated_recommendations' or "
            "'allow_recommending_already_measured' being False) "
            "or because all data points are marked as 'dont_recommend'."
        )

    # Get recommendations
    idxs = recommend(searchspace, candidates_comp, batch_quantity)
    rec = searchspace.discrete.exp_rep.loc[idxs, :]

    # Update metadata
    searchspace.discrete.metadata.loc[idxs, "was_recommended"] = True

    # Return recommendations
    return rec


@define
class Recommender(ABC):
    # TODO Docstrings missing

    # Class variables
    compatibility: ClassVar[SearchSpaceType]

    @abstractmethod
    def recommend(
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        pass


@define
class NonPredictiveRecommender(Recommender, ABC):
    def recommend(
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        """Recommend (a batch of) points in the searchspace.

        Depending on the type of the given searchspace, this method calls one of the
        corresponding private methods which implement the actual logic.

        Parameters
        ----------
        searchspace: SearchSpace
            The searchspace in which we are looking for a recommendation.
        batch_quantity: int, default = 1
            The batch quantity.
        train_x: pd.DataFrame, optional
            Training data. Since this recommender is non predictive, this is ignored.
        train_y: pd.DataFrame, optional
            See 'train_x'.
        allow_repeated_recommendates: bool, default = False
            Flag denoting whether repeated recommendations should be allowed. Only has
            an influence for discrete searchspaces.
        allow_recommending_already_measured: bool, default = True
            Flag denoting whether recommending already measured points should be
            allowed. Only has an influence for discrete searchspaces.

        Returns
        -------
        pd.DataFrame
            The recommendations
        """

        if searchspace.type == SearchSpaceType.DISCRETE:
            return select_candidates_and_recommend(
                searchspace,
                self._recommend_discrete,
                batch_quantity,
                allow_repeated_recommendations,
                allow_recommending_already_measured,
            )
        if searchspace.type == SearchSpaceType.CONTINUOUS:
            return self._recommend_continuous(
                searchspace=searchspace, batch_quantity=batch_quantity
            )
        return self._recommend_hybrid(
            searchspace=searchspace, batch_quantity=batch_quantity
        )

    def _recommend_discrete(
        self,
        searchspace: SearchSpace,
        candidates_comp: pd.DataFrame,
        batch_quantity: int,
    ) -> pd.Index:
        """If this method is not implemented by a children class, try to call
        _recommend_hybrid instead."""
        try:
            return self._recommend_hybrid(
                searchspace=searchspace,
                batch_quantity=batch_quantity,
                candidates_comp=candidates_comp,
            ).index
        except NotImplementedError as exc:
            raise NotImplementedError(
                """Hybrid recommender could not be used as fallback when trying to
                optimize a discrete space. This is probably due to your searchspace and
                recommender not being compatible. Please verify that your searchspace is
                purely discrete and that you are either using a discrete or hybrid
                recommender."""
            ) from exc

    def _recommend_continuous(
        self, searchspace: SearchSpace, batch_quantity: int
    ) -> pd.DataFrame:
        """If this method is not implemented by a children class, try to call
        _recommend_hybrid instead."""
        try:
            return self._recommend_hybrid(
                searchspace=searchspace, batch_quantity=batch_quantity
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                """Hybrid recommender could not be used as fallback when trying to
                optimize a continuous space. This is probably due to your searchspace
                and  recommender not being compatible. Please verify that your
                searchspace is purely continuous and that you are either using a
                continuous or hybrid recommender."""
            ) from exc

    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        batch_quantity: int,
        candidates_comp: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """In order to enable the fallback mechanism, it is possible to provide
        a DataFrame with candidates in computational representation."""
        raise NotImplementedError("Hybrid recommender is not implemented.")


# Register (un-)structure hooks
cattrs.register_unstructure_hook(Recommender, unstructure_base)
cattrs.register_structure_hook(Recommender, get_base_unstructure_hook(Recommender))
