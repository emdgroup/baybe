"""Kernel serialization tests."""

import hypothesis.strategies as st
from hypothesis import given

from baybe.kernels.base import Kernel
from tests.hypothesis_strategies.kernels import kernels, vanishing_kernels
from tests.serialization.utils import assert_roundtrip_consistency


@given(st.one_of(kernels(), vanishing_kernels()))
def test_roundtrip(kernel: Kernel):
    """A serialization roundtrip yields an equivalent object."""
    assert_roundtrip_consistency(kernel)
