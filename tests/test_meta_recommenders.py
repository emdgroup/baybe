"""Tests for meta recommenders."""

import pytest

from baybe.exceptions import NoRecommendersLeftError
from baybe.recommenders import (
    BotorchRecommender,
    FPSRecommender,
    RandomRecommender,
    SequentialMetaRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.recommenders.meta.sequential import StreamingSequentialMetaRecommender
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


@pytest.mark.parametrize(
    ("cls", "mode"),
    [
        (SequentialMetaRecommender, "raise"),
        (SequentialMetaRecommender, "reuse_last"),
        (SequentialMetaRecommender, "cyclic"),
        (StreamingSequentialMetaRecommender, None),
    ],
)
def test_sequential_meta_recommender(cls, mode):
    """The recommender provides its recommenders in the right order."""
    # Create meta recommender
    if cls == SequentialMetaRecommender:
        meta_recommender = SequentialMetaRecommender(
            recommenders=RECOMMENDERS, mode=mode
        )
    else:
        meta_recommender = StreamingSequentialMetaRecommender(
            recommenders=(r for r in RECOMMENDERS)  # <-- generator comprehension
        )

    training_size = 0

    # First iteration over provided recommender sequence
    for reference in RECOMMENDERS:
        training_size += 1

        # The returned recommender coincides with what was put in
        recommender = select_recommender(meta_recommender, training_size)
        assert recommender is reference

        # If the current recommender was not used (by recommending via the meta
        # recommender), a subsequent call returns the same recommender
        recommender = select_recommender(meta_recommender, training_size + 1)
        assert recommender is reference

        # Pretend the previous data size increase did not happen
        meta_recommender._n_last_measurements = training_size

        # Pretend the recommender was used
        meta_recommender._was_used = True

        # Selection with unchanged training size yields again the same recommender
        recommender = select_recommender(meta_recommender, training_size)
        assert recommender is reference

        # Selection with smaller training size raises an error
        with pytest.raises(
            RuntimeError,
            match=f"decreased from {training_size} to {training_size-1}",
        ):
            select_recommender(meta_recommender, training_size - 1)

    # For streaming recommenders, no second iteration is possible
    if cls == StreamingSequentialMetaRecommender:
        return

    # Second iteration over provided recommender sequence
    for cycled in RECOMMENDERS:
        training_size += 1

        if mode == "raise":
            # Requesting more batches than there are recommenders raises an error
            with pytest.raises(NoRecommendersLeftError):
                select_recommender(meta_recommender, training_size)

        elif mode == "reuse_last":
            # The last recommender is selected repeatedly
            recommender = select_recommender(meta_recommender, training_size)
            assert recommender == RECOMMENDERS[-1]

        elif mode == "cyclic":
            # The selection restarts from the first recommender
            recommender = select_recommender(meta_recommender, training_size)
            assert recommender == cycled

        # Pretend the recommender was used
        meta_recommender._was_used = True
