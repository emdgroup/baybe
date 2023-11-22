"""Tests for the strategies package."""

import pytest

from baybe.exceptions import NoRecommendersLeftError
from baybe.recommenders import (
    FPSRecommender,
    RandomRecommender,
    SequentialGreedyRecommender,
)
from baybe.strategies.composite import (
    SequentialStrategy,
    StreamingSequentialStrategy,
    TwoPhaseStrategy,
)
from tests.conftest import select_recommender

RECOMMENDERS = [RandomRecommender(), FPSRecommender(), SequentialGreedyRecommender()]


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
