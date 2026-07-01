"""Bayesian recommenders."""

from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.recommenders.pure.bayesian.core import BayesianRecommender

__all__ = [
    "BayesianRecommender",
    "BotorchRecommender",
]
