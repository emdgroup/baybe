# pylint: disable=missing-module-docstring, missing-function-docstring

from baybe.strategies.strategy import Strategy


def test_strategy_serialization(strategy):
    string = strategy.to_json()
    strategy2 = Strategy.from_json(string)
    assert strategy == strategy2
