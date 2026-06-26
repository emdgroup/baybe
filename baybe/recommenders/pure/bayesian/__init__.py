"""Bayesian recommenders."""

from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender

__all__ = [
    "BayesianRecommender",
    "BotorchRecommender",
]
