# pylint: disable=missing-module-docstring, missing-function-docstring

import json

from baybe.targets import Objective


def test_objective_serialization(objective):
    string = json.dumps(objective.to_dict())
    objective2 = Objective.from_dict(json.loads(string))
    assert objective == objective2
