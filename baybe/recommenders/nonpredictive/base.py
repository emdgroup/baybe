"""Base class for all nonpredictive recommenders."""

from abc import ABC

from attrs import define

from baybe.recommenders.base import Recommender


@define
class NonPredictiveRecommender(Recommender, ABC):
    """Abstract base class for recommenders that are non-predictive."""
