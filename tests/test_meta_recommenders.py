"""Tests for meta recommenders."""

import pytest

from baybe.exceptions import NoRecommendersLeftError
from baybe.recommenders import (
    BotorchRecommender,
    FPSRecommender,
    RandomRecommender,
    SequentialMetaRecommender,
    StreamingSequentialMetaRecommender,
    TwoPhaseMetaRecommender,
)
from tests.conftest import select_recommender

RECOMMENDERS = [RandomRecommender(), FPSRecommender(), BotorchRecommender()]


def test_twophase_meta_recommender():
    """The recommender switches the recommender at the requested point."""
    initial_recommender = RandomRecommender()
    subsequent_recommender = RandomRecommender()
    switch_after = 3
    recommender = TwoPhaseMetaRecommender(
        initial_recommender=initial_recommender,
        recommender=subsequent_recommender,
        switch_after=switch_after,
    )
    for training_size in range(6):
        rec = select_recommender(recommender, training_size)
        target = (
            initial_recommender
            if training_size < switch_after
            else subsequent_recommender
        )
        assert rec is target


@pytest.mark.parametrize("mode", ["raise", "reuse_last", "cyclic"])
@pytest.mark.parametrize("recommenders", [RECOMMENDERS])
def test_sequential_meta_recommender(recommenders, mode):
    """The recommender provides its recommenders in the right order."""
    meta_recommender = SequentialMetaRecommender(recommenders=recommenders, mode=mode)
    training_size = 0

    # First iteration over provided recommender sequence
    for reference in recommenders:
        training_size += 1

        # The returned recommender coincides with what was put in
        recommender = select_recommender(meta_recommender, training_size)
        assert recommender is reference

        # Selection with unchanged training size yields again the same recommender
        recommender = select_recommender(meta_recommender, training_size)
        assert recommender is reference

        # Selection with smaller training size raises an error
        with pytest.raises(RuntimeError):
            select_recommender(meta_recommender, training_size - 1)

    # Second iteration over provided recommender sequence
    for cycled in recommenders:
        training_size += 1

        if mode == "raise":
            # Requesting more batches than there are recommenders raises an error
            with pytest.raises(NoRecommendersLeftError):
                select_recommender(meta_recommender, training_size)

        elif mode == "reuse_last":
            # The last recommender is selected repeatedly
            recommender = select_recommender(meta_recommender, training_size)
            assert recommender == recommenders[-1]

        elif mode == "cyclic":
            # The selection restarts from the first recommender
            recommender = select_recommender(meta_recommender, training_size)
            assert recommender == cycled


@pytest.mark.parametrize(
    "recommenders",
    [
        RECOMMENDERS,  # list
        (rec for rec in RECOMMENDERS),  # generator
    ],
)
def test_streaming_sequential_meta_recommender(recommenders):
    """The recommender provides its recommenders in the right order."""
    meta_recommender = StreamingSequentialMetaRecommender(recommenders=recommenders)
    training_size = 0

    for reference in RECOMMENDERS:
        training_size += 1

        # The returned recommender coincides with what was put in
        recommender = select_recommender(meta_recommender, training_size)
        assert recommender is reference

        # Selection with unchanged training size yields again the same recommender
        recommender = select_recommender(meta_recommender, training_size)
        assert recommender is reference

        # Selection with smaller training size raises an error
        with pytest.raises(RuntimeError):
            select_recommender(meta_recommender, training_size - 1)
