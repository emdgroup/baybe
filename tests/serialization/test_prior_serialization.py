"""Prior serialization tests."""

from hypothesis import given

from baybe.priors.base import Prior
from tests.hypothesis_strategies.priors import priors
from tests.serialization.utils import assert_roundtrip_consistency


@given(priors(gpytorch_only=False))
def test_roundtrip(prior: Prior):
    """A serialization roundtrip yields an equivalent object."""
    assert_roundtrip_consistency(prior)
