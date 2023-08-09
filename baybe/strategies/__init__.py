"""Recommendation functionality."""

from baybe.strategies.bayesian import (
    NaiveHybridRecommender,
    SequentialGreedyRecommender,
)
from baybe.strategies.clustering import (
    GaussianMixtureClusteringRecommender,
    KMeansClusteringRecommender,
)
from baybe.strategies.sampling import FPSRecommender, RandomRecommender
from baybe.strategies.strategy import Strategy


__all__ = [
    "FPSRecommender",
    "GaussianMixtureClusteringRecommender",
    "KMeansClusteringRecommender",
    "NaiveHybridRecommender",
    "RandomRecommender",
    "SequentialGreedyRecommender",
    "Strategy",
]

try:
    from baybe.strategies.clustering import PAMClusteringRecommender

    __all__.append("PAMClusteringRecommender")
except ImportError:
    pass
