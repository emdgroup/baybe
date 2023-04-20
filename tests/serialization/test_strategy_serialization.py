# pylint: disable=missing-module-docstring, missing-function-docstring

import json

from baybe.strategies.strategy import Strategy


def test_objective_serialization(strategy):
    string = json.dumps(strategy.to_dict())
    strategy2 = Strategy.from_dict(json.loads(string))
    assert strategy == strategy2
