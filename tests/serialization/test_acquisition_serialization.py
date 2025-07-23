"""Acquisition function serialization tests."""

import hypothesis.strategies as st
from hypothesis import given

from tests.hypothesis_strategies.acquisition import acquisition_functions
from tests.serialization.utils import assert_roundtrip_consistency


@given(acquisition_functions)
def test_roundtrip(acqf: st.SearchStrategy):
    """A serialization roundtrip yields an equivalent object."""
    assert_roundtrip_consistency(acqf)
