"""Objective serialization tests."""

import hypothesis.strategies as st
import pytest
from hypothesis import given
from pytest import param

from tests.hypothesis_strategies.objectives import (
    desirability_objectives,
    pareto_objectives,
    single_target_objectives,
)
from tests.serialization.utils import assert_roundtrip_consistency


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
