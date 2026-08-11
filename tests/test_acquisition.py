"""Acquisition function tests."""

from hypothesis import given

from baybe.recommenders import BayesianRecommender
from tests.hypothesis_strategies.acquisition import acquisition_functions


@given(acquisition_functions)
def test_acqfs(acqf):
    """Test all acquisition functions with sequential greedy recommender."""
    BayesianRecommender(acquisition_function=acqf)
