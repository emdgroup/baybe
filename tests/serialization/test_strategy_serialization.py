# pylint: disable=missing-module-docstring, missing-function-docstring
"""Test serialization of strategies."""

import pytest

from baybe.recommenders import RandomRecommender
from baybe.strategies import (
    SequentialStrategy,
    StreamingSequentialStrategy,
    TwoPhaseStrategy,
)
from baybe.strategies.base import Strategy


@pytest.mark.parametrize(
    "strategy",
    [TwoPhaseStrategy(), SequentialStrategy(recommenders=[RandomRecommender()])],
)
def test_strategy_serialization(strategy):
    """Roundtrip serialization of strategies."""
    string = strategy.to_json()
    strategy2 = Strategy.from_json(string)
    assert strategy == strategy2


def test_unsupported_serialization():
    """Attempting to serialize an unserializable strategy should raise an error."""
    strategy = StreamingSequentialStrategy(
        recommenders=(rec for rec in [RandomRecommender()])
    )
    with pytest.raises(NotImplementedError):
        strategy.to_json()
