"""Base classes for all recommenders."""

from abc import ABC
from typing import Callable, ClassVar, Optional, Protocol

import cattrs
import pandas as pd
from attrs import define, field

from baybe.exceptions import NotEnoughPointsLeftError
from baybe.recommenders.deprecation import structure_recommender_protocol
from baybe.searchspace import (
    SearchSpace,
    SearchSpaceType,
    SubspaceDiscrete,
)
from baybe.serialization import (
    converter,
    unstructure_base,
)


def _select_candidates_and_recommend(
    searchspace: SearchSpace,
    recommend_discrete: Callable[[SubspaceDiscrete, pd.DataFrame, int], pd.Index],
    batch_size: int = 1,
    allow_repeated_recommendations: bool = False,
    allow_recommending_already_measured: bool = True,
) -> pd.DataFrame:
    """Select candidates in a discrete search space and recommend them.

    This function is a workaround as this functionality is required for all purely
    discrete recommenders and avoids the introduction of complicate class hierarchies.
    It is also used to select candidates in the discrete part of hybrid search spaces,
    ignoring the continuous part.

    Args:
        searchspace: The search space.
        recommend_discrete: The Callable representing the discrete recommendation
            function.
        batch_size: The chosen batch size.
        allow_repeated_recommendations: Allow to make recommendations that were already
            recommended earlier.
        allow_recommending_already_measured: Allow to output recommendations that were
            measured previously.

    Returns:
        The recommendation in experimental representation.

    Raises:
        NotEnoughPointsLeftError: If there are fewer than ``batch_size`` points
            left for potential recommendation.
    """
    # IMPROVE: See if the there is a more elegant way to share this functionality
    #   among all purely discrete recommenders (without introducing complicates class
    #   hierarchies).

    # Get discrete candidates. The metadata flags are ignored if the search space
    # has a continuous component.
    _, candidates_comp = searchspace.discrete.get_candidates(
        allow_repeated_recommendations=allow_repeated_recommendations
        or not searchspace.continuous.is_empty,
        allow_recommending_already_measured=allow_recommending_already_measured
        or not searchspace.continuous.is_empty,
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
    idxs = recommend_discrete(searchspace.discrete, candidates_comp, batch_size)
    rec = searchspace.discrete.exp_rep.loc[idxs, :]

    # Update metadata
    searchspace.discrete.metadata.loc[idxs, "was_recommended"] = True

    # Return recommendations
    return rec


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
