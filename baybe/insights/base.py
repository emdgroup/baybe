"""Base class for all insights."""

from abc import ABC

import pandas as pd

from baybe import Campaign
from baybe._optional.info import INSIGHTS_INSTALLED
from baybe.objectives.base import Objective
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.searchspace import SearchSpace

if INSIGHTS_INSTALLED:
    pass


class Insight(ABC):
    """Base class for all insights."""

    def __init__(self, surrogate):
        self.surrogate = surrogate

    @classmethod
    def from_campaign(cls, campaign: Campaign):
        """Create an insight from a campaign."""
        return cls(campaign.get_surrogate())

    @classmethod
    def from_recommender(
        cls,
        recommender: BayesianRecommender,
        searchspace: SearchSpace,
        objective: Objective,
        bg_data: pd.DataFrame,
    ):
        """Create an insight from a recommender."""
        if not hasattr(recommender, "get_surrogate"):
            raise ValueError(
                "The provided recommender does not provide a surrogate model."
            )
        surrogate_model = recommender.get_surrogate(searchspace, objective, bg_data)

        return cls(
            surrogate_model,
        )
