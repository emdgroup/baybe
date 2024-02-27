"""Base protocol for all recommenders."""

from typing import Optional, Protocol

import cattrs
import pandas as pd

from baybe.recommenders.deprecation import structure_recommender_protocol
from baybe.searchspace import SearchSpace
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
