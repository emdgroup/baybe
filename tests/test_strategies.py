"""Tests for the strategies package."""

from contextlib import nullcontext

import pytest

from baybe.exceptions import NoRecommendersLeftError
from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders import (
    FPSRecommender,
    RandomRecommender,
    SequentialGreedyRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.strategies.base import Strategy
from baybe.strategies.composite import SequentialStrategy, StreamingSequentialStrategy

RECOMMENDERS = [RandomRecommender(), FPSRecommender(), SequentialGreedyRecommender()]


@pytest.mark.parametrize("reuse_last", [False, True])
@pytest.mark.parametrize("recommenders", [RECOMMENDERS])
def test_sequential_strategy(recommenders, reuse_last):
    """The strategy should provide its recommenders in the right order."""
    parameters = [NumericalDiscreteParameter(name="test", values=[0, 1])]
    searchspace = SearchSpace.from_product(parameters)
    strategy = SequentialStrategy(recommenders=recommenders, reuse_last=reuse_last)

    # The returned recommenders should coincide with what was put in
    for reference in recommenders[:-1]:
        recommender = strategy.select_recommender(searchspace)
        assert recommender is reference

    # After serialization, identity does not hold but equality should
    strategy = Strategy.from_json(strategy.to_json())
    rec2 = strategy.select_recommender(searchspace)
    assert rec2 == recommenders[2]

    # Requesting more batches than there are recommenders should raise an error
    with nullcontext() if reuse_last else pytest.raises(NoRecommendersLeftError):
        strategy.select_recommender(searchspace)


@pytest.mark.parametrize(
    "recommenders",
    [
        RECOMMENDERS,  # list
        (rec for rec in RECOMMENDERS),  # generator
    ],
)
def test_streaming_sequential_strategy(recommenders):
    """The strategy should provide its recommenders in the right order."""
    parameters = [NumericalDiscreteParameter(name="test", values=[0, 1])]
    searchspace = SearchSpace.from_product(parameters)
    strategy = StreamingSequentialStrategy(recommenders=recommenders)

    # The returned recommenders should coincide with what was put in
    for reference in RECOMMENDERS:
        recommender = strategy.select_recommender(searchspace)
        assert recommender is reference

    # Requesting more batches than there are recommenders should raise an error
    with pytest.raises(NoRecommendersLeftError):
        strategy.select_recommender(searchspace)
