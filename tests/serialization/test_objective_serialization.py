"""Test serialization of objectives."""

import hypothesis.strategies as st
import pytest
from hypothesis import given
from pytest import param

from baybe.objectives.base import Objective
from tests.hypothesis_strategies.objectives import (
    desirability_objectives,
    single_target_objectives,
)


@pytest.mark.parametrize(
    "objective_strategy",
    [
        param(single_target_objectives(), id="SingleTargetObjective"),
        param(desirability_objectives(), id="DesirabilityObjective"),
    ],
)
@given(data=st.data())
def test_objective_roundtrip(objective_strategy, data):
    """A serialization roundtrip yields an equivalent object."""
    objective = data.draw(objective_strategy)
    string = objective.to_json()
    objective2 = Objective.from_json(string)
    assert objective == objective2, (objective, objective2)
