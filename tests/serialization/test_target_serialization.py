"""Test serialization of targets."""

import pytest
from hypothesis import given

from baybe.targets.base import Target
from baybe.targets.numerical import NumericalTarget
from baybe.transformations.core import ClampingTransformation
from tests.serialization.test_objective_serialization import (
    _get_involved_transformations,
)

from ..hypothesis_strategies.targets import targets


@given(targets)
def test_target_roundtrip(target: Target):
    """A serialization roundtrip yields an equivalent object."""
    if isinstance(target, NumericalTarget) and any(
        isinstance(t, ClampingTransformation)
        for t in _get_involved_transformations(target)
    ):
        pytest.xfail(
            reason=(
                "Serialization of clamping transformations is not yet supported. "
                "Needs https://github.com/emdgroup/baybe/pull/577"
            )
        )
    string = target.to_json()
    target2 = Target.from_json(string)
    assert target == target2, (target, target2)
