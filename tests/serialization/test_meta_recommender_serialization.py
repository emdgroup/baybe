"""Meta recommender serialization tests."""

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
from tests.serialization.utils import assert_roundtrip_consistency, roundtrip

# Create some recommenders of different class for better differentiation after roundtrip
RECOMMENDERS = [RandomRecommender(), FPSRecommender()]
assert len(RECOMMENDERS) == len({rec.__class__.__name__ for rec in RECOMMENDERS})


@pytest.mark.parametrize(
    "recommender",
    [
        TwoPhaseMetaRecommender(),
        SequentialMetaRecommender(recommenders=[RandomRecommender()]),
    ],
)
def test_roundtrip(recommender: MetaRecommender):
    """A serialization roundtrip yields an equivalent object."""
    assert_roundtrip_consistency(recommender)


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
    if isinstance(recommender, SequentialMetaRecommender):
        recommender._was_used = True
    rec = select_recommender(recommender, 1)
    assert rec is RECOMMENDERS[1]

    # After serialization, identity no longer holds but equality does
    recommender2 = roundtrip(recommender)
    assert recommender2 == recommender
    rec = select_recommender(recommender2, 1)
    assert rec == RECOMMENDERS[1]
