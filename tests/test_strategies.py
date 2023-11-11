"""Tests for the strategies package."""

from contextlib import nullcontext
from typing import Tuple

import numpy as np
import pandas as pd
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
from baybe.strategies.composite import (
    SequentialStrategy,
    StreamingSequentialStrategy,
    TwoPhaseStrategy,
)

RECOMMENDERS = [RandomRecommender(), FPSRecommender(), SequentialGreedyRecommender()]


def get_dummy_training_data(length: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create column-less input and target dataframes of specified length."""
    df = pd.DataFrame(np.empty((length, 0)))
    return df, df


@pytest.fixture(name="searchspace")
def get_dummy_searchspace() -> SearchSpace:
    """Create a dummy searchspace whose actual content is irrelevant."""
    parameters = [NumericalDiscreteParameter(name="test", values=[0, 1])]
    return SearchSpace.from_product(parameters)


def test_twophase_strategy(searchspace):
    """The strategy switches the recommender at the requested point."""
    initial_recommender = RandomRecommender()
    recommender = RandomRecommender()
    switch_after = 3
    strategy = TwoPhaseStrategy(
        initial_recommender=initial_recommender,
        recommender=recommender,
        switch_after=switch_after,
    )
    for size in range(6):
        train_x, train_y = get_dummy_training_data(size)
        rec = strategy.select_recommender(searchspace, train_x=train_x, train_y=train_y)
        target = initial_recommender if size < switch_after else recommender
        assert rec is target


@pytest.mark.parametrize("reuse_last", [False, True])
@pytest.mark.parametrize("recommenders", [RECOMMENDERS])
def test_sequential_strategy(searchspace, recommenders, reuse_last):
    """The strategy should provide its recommenders in the right order."""
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
def test_streaming_sequential_strategy(searchspace, recommenders):
    """The strategy should provide its recommenders in the right order."""
    strategy = StreamingSequentialStrategy(recommenders=recommenders)

    # The returned recommenders should coincide with what was put in
    for reference in RECOMMENDERS:
        recommender = strategy.select_recommender(searchspace)
        assert recommender is reference

    # Requesting more batches than there are recommenders should raise an error
    with pytest.raises(NoRecommendersLeftError):
        strategy.select_recommender(searchspace)
