"""Test serialization of objectives."""

from itertools import chain

import hypothesis.strategies as st
import pytest
from hypothesis import given
from pytest import param

from baybe.objectives.base import Objective
from baybe.targets.base import Target
from baybe.transformations import (
    ChainedTransformation,
    ClampingTransformation,
    Transformation,
)
from tests.hypothesis_strategies.objectives import (
    desirability_objectives,
    pareto_objectives,
    single_target_objectives,
)


def _get_involved_transformations(target: Target) -> list[Transformation]:
    """Get all transformations involved in a target."""
    match t := target._transformation:
        case ChainedTransformation():
            return t.transformations
        case _:
            return [t]


@pytest.mark.parametrize(
    "objective_strategy",
    [
        param(single_target_objectives(), id="SingleTargetObjective"),
        param(desirability_objectives(), id="DesirabilityObjective"),
        param(pareto_objectives(), id="ParetoObjective"),
    ],
)
@given(data=st.data())
def test_objective_roundtrip(objective_strategy, data):
    """A serialization roundtrip yields an equivalent object."""
    objective = data.draw(objective_strategy)
    transformations = chain.from_iterable(
        _get_involved_transformations(t) for t in objective.targets
    )

    if any(isinstance(t, ClampingTransformation) for t in transformations):
        pytest.xfail(
            reason=(
                "Serialization of clamping transformations is not yet supported. "
                "Needs https://github.com/emdgroup/baybe/pull/577"
            )
        )
    string = objective.to_json()
    objective2 = Objective.from_json(string)
    assert objective == objective2, (objective, objective2)
