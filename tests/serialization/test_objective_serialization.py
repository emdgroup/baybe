"""Objective serialization tests."""

import hypothesis.strategies as st
import pytest
from hypothesis import given
from pytest import param

from baybe.targets.base import Target
from baybe.transformations import (
    ChainedTransformation,
    Transformation,
)
from tests.hypothesis_strategies.objectives import (
    desirability_objectives,
    pareto_objectives,
    single_target_objectives,
)
from tests.serialization.utils import assert_roundtrip_consistency


def _get_involved_transformations(target: Target) -> list[Transformation]:
    """Get all transformations involved in a target."""
    match t := target.transformation:
        case ChainedTransformation():
            return t.transformations
        case _:
            return [t]


@pytest.mark.parametrize(
    "strategy",
    [
        param(single_target_objectives(), id="SingleTargetObjective"),
        param(desirability_objectives(), id="DesirabilityObjective"),
        param(pareto_objectives(), id="ParetoObjective"),
    ],
)
@given(data=st.data())
def test_roundtrip(strategy: st.SearchStrategy, data: st.DataObject):
    """A serialization roundtrip yields an equivalent object."""
    objective = data.draw(strategy)
    assert_roundtrip_consistency(objective)
