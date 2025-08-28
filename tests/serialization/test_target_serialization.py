"""Test serialization of targets."""

from hypothesis import given

from baybe.targets.base import Target

from ..hypothesis_strategies.targets import targets


@given(targets)
def test_target_roundtrip(target: Target):
    """A serialization roundtrip yields an equivalent object."""
    string = target.to_json()
    target2 = Target.from_json(string)
    assert target == target2, (target, target2)
