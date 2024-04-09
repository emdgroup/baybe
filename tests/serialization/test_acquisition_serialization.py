"""Test serialization of acquisition functions."""

from hypothesis import given

from baybe.acquisition.base import AcquisitionFunction
from tests.hypothesis_strategies.acquisition import random_acqfs


@given(random_acqfs)
def test_acqf_roundtrip(acqf):
    """A serialization roundtrip yields an equivalent object."""
    string = acqf.to_json()
    acqf2 = AcquisitionFunction.from_json(string)
    assert acqf == acqf2, (acqf, acqf2)
