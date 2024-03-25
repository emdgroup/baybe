"""Test serialization of meta recommenders."""

import pytest

from baybe.recommenders import (
    FPSRecommender,
    RandomRecommender,
    SequentialMetaRecommender,
    StreamingSequentialMetaRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.recommenders.meta.base import MetaRecommender
from tests.conftest import select_recommender

# Create some recommenders of different class for better differentiation after roundtrip
RECOMMENDERS = [RandomRecommender(), FPSRecommender()]
assert len(RECOMMENDERS) == len({rec.__class__.__name__ for rec in RECOMMENDERS})


def roundtrip(recommender: MetaRecommender) -> MetaRecommender:
    """Roundtrip serialization."""
    string = recommender.to_json()
    return MetaRecommender.from_json(string)


@pytest.mark.parametrize(
    "recommender",
    [
        TwoPhaseMetaRecommender(),
        SequentialMetaRecommender(recommenders=[RandomRecommender()]),
    ],
)
def test_meta_recommender_serialization(recommender):
    """Roundtrip serialization of meta recommenders."""
    assert recommender == roundtrip(recommender)


def test_unsupported_serialization():
    """Attempting to serialize an unserializable recommender should raise an error."""
    recommender = StreamingSequentialMetaRecommender(
        recommenders=(rec for rec in [RandomRecommender()])
    )
    with pytest.raises(NotImplementedError):
        recommender.to_json()


@pytest.mark.parametrize(
    "recommender",
    [
        TwoPhaseMetaRecommender(
            initial_recommender=RECOMMENDERS[0],
            recommender=RECOMMENDERS[1],
            switch_after=1,
        ),
        SequentialMetaRecommender(recommenders=RECOMMENDERS),
    ],
)
def test_meta_recommender_state_serialization(recommender):
    """Roundtrip-serialized meta recommenders keep their internal states."""
    # Before serialization, identity must hold
    rec = select_recommender(recommender, 0)
    assert rec is RECOMMENDERS[0]
    rec = select_recommender(recommender, 1)
    assert rec is RECOMMENDERS[1]

    # After serialization, identity no longer holds but equality does
    recommender = roundtrip(recommender)
    rec = select_recommender(recommender, 1)
    assert rec == RECOMMENDERS[1]
