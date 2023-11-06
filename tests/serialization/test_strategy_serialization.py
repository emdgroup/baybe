# pylint: disable=missing-module-docstring, missing-function-docstring
"""Test serialization of strategies."""

from baybe.strategies.base import Strategy


def test_strategy_serialization(strategy):
    string = strategy.to_json()
    strategy2 = Strategy.from_json(string)
    assert strategy == strategy2
