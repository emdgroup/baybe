"""Acquisition function tests."""

from hypothesis import given

from baybe.recommenders import SequentialGreedyRecommender

from .hypothesis_strategies.acquisition import random_acqfs


@given(random_acqfs)
def test_acqfs(acqf):
    """Test all acquisition functions with sequential greedy recommender."""
    SequentialGreedyRecommender(acqf=acqf)
