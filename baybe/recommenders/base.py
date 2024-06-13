"""Base protocol for all recommenders."""

from typing import Protocol, runtime_checkable

import cattrs
import pandas as pd

from baybe.objectives.base import Objective
from baybe.recommenders.deprecation import structure_recommender_protocol
from baybe.searchspace import SearchSpace
from baybe.serialization import converter, unstructure_base


@runtime_checkable
class RecommenderProtocol(Protocol):
    """Type protocol specifying the interface recommenders need to implement."""

    def recommend(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None,
        measurements: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Recommend a batch of points from the given search space.

        Args:
            batch_size: The number of points to be recommended.
            searchspace: The search space from which to recommend the points.
            objective: An optional objective to be optimized.
            measurements: Optional experimentation data that can be used for model
                training. The data is to be provided in "experimental representation":
                It needs to contain one column for each parameter spanning the search
                space (column name matching the parameter name) and one column for each
                target tracked by the objective (column name matching the target name).
                Each row corresponds to one conducted experiment, where the parameter
                columns define the experimental setting and the target columns report
                the measured outcomes.

        Returns:
            A dataframe containing the recommendations in experimental representation
            as individual rows.
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
            acquisition_function_cls=cattrs.override(omit=True),
        ),
    ),
)
converter.register_structure_hook(RecommenderProtocol, structure_recommender_protocol)
