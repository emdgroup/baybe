"""Base class for all insights."""

from __future__ import annotations

from abc import ABC

import pandas as pd
from attrs import define, field

from baybe import Campaign
from baybe.objectives.base import Objective
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.searchspace import SearchSpace
from baybe.surrogates.base import SurrogateProtocol


@define
class Insight(ABC):
    """Base class for all insights."""

    surrogate: SurrogateProtocol = field()
    """The surrogate model that is supposed bo be analyzed."""

    @classmethod
    def from_campaign(cls, campaign: Campaign) -> Insight:
        """Create an insight from a campaign.

        Args:
            campaign: A baybe Campaign object.

        Returns:
            The Insight object.
        """
        return cls(campaign.get_surrogate())

    @classmethod
    def from_recommender(
        cls,
        recommender: BayesianRecommender,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> Insight:
        """Create an insight from a recommender.

        Args:
            recommender: A model-based recommender.
            searchspace: The search space used for recommendations.
            objective: The objective of the recommendation.
            measurements: The measurements in experimental representation.

        Returns:
            The Insight object.

        Raises:
            ValueError: If the provided recommender is not surrogate-based.
        """
        if not hasattr(recommender, "get_surrogate"):
            raise ValueError(
                f"The provided recommender of type '{recommender.__class__.__name__}' "
                f"does not provide a surrogate model."
            )
        surrogate_model = recommender.get_surrogate(
            searchspace, objective, measurements
        )

        return cls(surrogate_model)
