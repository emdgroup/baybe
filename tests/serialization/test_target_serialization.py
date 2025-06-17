"""Target serialization tests."""

from hypothesis import given

from baybe.targets.base import Target
from tests.hypothesis_strategies.targets import targets
from tests.serialization.utils import assert_roundtrip_consistency


@given(targets)
def test_roundtrip(target: Target):
    """A serialization roundtrip yields an equivalent object."""
    assert_roundtrip_consistency(target)
