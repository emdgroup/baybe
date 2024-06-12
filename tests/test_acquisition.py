"""Acquisition function tests."""

from hypothesis import given

from baybe.recommenders import BotorchRecommender

from .hypothesis_strategies.acquisition import acquisition_functions


@given(acquisition_functions)
def test_acqfs(acqf):
    """Test all acquisition functions with sequential greedy recommender."""
    BotorchRecommender(acquisition_function=acqf)
