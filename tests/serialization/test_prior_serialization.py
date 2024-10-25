"""Test serialization of priors."""

from hypothesis import given

from baybe.priors.base import Prior
from tests.hypothesis_strategies.priors import priors


@given(priors(gpytorch_only=False))
def test_prior_kernel_roundtrip(prior: Prior):
    string = prior.to_json()
    prior2 = Prior.from_json(string)
    assert prior == prior2, (prior, prior2)
