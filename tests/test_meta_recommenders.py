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


@pytest.mark.parametrize("remain_switched", [False, True])
def test_twophase_meta_recommender(remain_switched):
    """The recommender switches the recommender at the requested point and
    remains / reverts the switch depending on the configuration."""  # noqa
    # Cross the switching point forwards and backwards
    switch_after = 3
    training_sizes = [2, 3, 2]

    # Recommender objects
    initial_recommender = RandomRecommender()
    subsequent_recommender = RandomRecommender()
    recommender = TwoPhaseMetaRecommender(
        initial_recommender=initial_recommender,
        recommender=subsequent_recommender,
        switch_after=switch_after,
        remain_switched=remain_switched,
    )

    # Query the meta recommender
    switch_point_passed = False
    for n_data in training_sizes:
        rec = select_recommender(recommender, n_data)
        target = (
            subsequent_recommender
            if (n_data >= switch_after) or (switch_point_passed and remain_switched)
            else initial_recommender
        )
        if not switch_point_passed and n_data >= switch_after:
            switch_point_passed = True
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
