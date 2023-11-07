# pylint: disable=missing-module-docstring, missing-function-docstring
"""Test serialization of objectives."""

from baybe.objective import Objective


def test_objective_serialization(objective):
    string = objective.to_json()
    objective2 = Objective.from_json(string)
    assert objective == objective2
