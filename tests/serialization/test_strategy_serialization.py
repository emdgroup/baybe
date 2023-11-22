"""Test serialization of strategies."""

import pytest

from baybe.recommenders import FPSRecommender, RandomRecommender
from baybe.strategies import (
    SequentialStrategy,
    StreamingSequentialStrategy,
    TwoPhaseStrategy,
)
from baybe.strategies.base import Strategy
from tests.conftest import select_recommender

# Create some recommenders of different class for better differentiation after roundtrip
RECOMMENDERS = [RandomRecommender(), FPSRecommender()]
assert len(RECOMMENDERS) == len(set(rec.__class__.__name__ for rec in RECOMMENDERS))


def roundtrip(strategy: Strategy) -> Strategy:
    """Roundtrip serialization."""
    string = strategy.to_json()
    return Strategy.from_json(string)


@pytest.mark.parametrize(
    "strategy",
    [TwoPhaseStrategy(), SequentialStrategy(recommenders=[RandomRecommender()])],
)
def test_strategy_serialization(strategy):
    """Roundtrip serialization of strategies."""
    assert strategy == roundtrip(strategy)


def test_unsupported_serialization():
    """Attempting to serialize an unserializable strategy should raise an error."""
    strategy = StreamingSequentialStrategy(
        recommenders=(rec for rec in [RandomRecommender()])
    )
    with pytest.raises(NotImplementedError):
        strategy.to_json()


@pytest.mark.parametrize(
    "strategy",
    [
        TwoPhaseStrategy(
            initial_recommender=RECOMMENDERS[0],
            recommender=RECOMMENDERS[1],
            switch_after=1,
        ),
        SequentialStrategy(recommenders=RECOMMENDERS),
    ],
)
def test_strategy_state_serialization(strategy):
    """Roundtrip-serialized strategies keep their internal states."""
    # Before serialization, identity must hold
    rec = select_recommender(strategy, 0)
    assert rec is RECOMMENDERS[0]
    rec = select_recommender(strategy, 1)
    assert rec is RECOMMENDERS[1]

    # After serialization, identity no longer holds but equality does
    strategy = roundtrip(strategy)
    rec = select_recommender(strategy, 1)
    assert rec == RECOMMENDERS[1]
