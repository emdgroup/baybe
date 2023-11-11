"""Tests for the strategies package."""

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
from baybe.recommenders.base import Recommender
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


def get_dummy_searchspace() -> SearchSpace:
    """Create a dummy searchspace whose actual content is irrelevant."""
    parameters = [NumericalDiscreteParameter(name="test", values=[0, 1])]
    return SearchSpace.from_product(parameters)


def select_recommender(strategy: Strategy, training_size: int) -> Recommender:
    """Select a recommender for given training dataset size."""
    searchspace = get_dummy_searchspace()
    df_x, df_y = get_dummy_training_data(training_size)
    return strategy.select_recommender(searchspace, train_x=df_x, train_y=df_y)


def test_twophase_strategy():
    """The strategy switches the recommender at the requested point."""
    initial_recommender = RandomRecommender()
    recommender = RandomRecommender()
    switch_after = 3
    strategy = TwoPhaseStrategy(
        initial_recommender=initial_recommender,
        recommender=recommender,
        switch_after=switch_after,
    )
    for training_size in range(6):
        rec = select_recommender(strategy, training_size)
        target = initial_recommender if training_size < switch_after else recommender
        assert rec is target


@pytest.mark.parametrize("mode", ["raise", "reuse_last", "cyclic"])
@pytest.mark.parametrize("recommenders", [RECOMMENDERS])
def test_sequential_strategy(recommenders, mode):
    """The strategy provides its recommenders in the right order."""
    strategy = SequentialStrategy(recommenders=recommenders, mode=mode)
    training_size = 0

    # First iteration over provided recommender sequence
    for reference in recommenders:
        training_size += 1

        # The returned recommender coincides with what was put in
        recommender = select_recommender(strategy, training_size)
        assert recommender is reference

        # Selection with unchanged training size yields again the same recommender
        recommender = select_recommender(strategy, training_size)
        assert recommender is reference

        # Selection with smaller training size raises an error
        with pytest.raises(RuntimeError):
            select_recommender(strategy, training_size - 1)

    # Second iteration over provided recommender sequence
    for cycled in recommenders:
        training_size += 1

        if mode == "raise":
            # Requesting more batches than there are recommenders raises an error
            with pytest.raises(NoRecommendersLeftError):
                select_recommender(strategy, training_size)

        elif mode == "reuse_last":
            # The last recommender is selected repeatedly
            recommender = select_recommender(strategy, training_size)
            assert recommender == recommenders[-1]

        elif mode == "cyclic":
            # The selection restarts from the first recommender
            recommender = select_recommender(strategy, training_size)
            assert recommender == cycled


@pytest.mark.parametrize(
    "recommenders",
    [
        RECOMMENDERS,  # list
        (rec for rec in RECOMMENDERS),  # generator
    ],
)
def test_streaming_sequential_strategy(recommenders):
    """The strategy provides its recommenders in the right order."""
    strategy = StreamingSequentialStrategy(recommenders=recommenders)
    training_size = 0

    for reference in RECOMMENDERS:
        training_size += 1

        # The returned recommender coincides with what was put in
        recommender = select_recommender(strategy, training_size)
        assert recommender is reference

        # Selection with unchanged training size yields again the same recommender
        recommender = select_recommender(strategy, training_size)
        assert recommender is reference

        # Selection with smaller training size raises an error
        with pytest.raises(RuntimeError):
            select_recommender(strategy, training_size - 1)
