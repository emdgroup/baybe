"""Base class for all nonpredictive recommenders."""

from abc import ABC

from attrs import define

from baybe.recommenders.pure.base import PureRecommender


@define
class NonPredictiveRecommender(PureRecommender, ABC):
    """Abstract base class for all nonpredictive recommenders."""
