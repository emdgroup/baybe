"""Kernel serialization tests."""

from hypothesis import given

from baybe.kernels.base import Kernel
from tests.hypothesis_strategies.kernels import kernels
from tests.serialization.utils import assert_roundtrip_consistency


@given(kernels())
def test_roundtrip(kernel: Kernel):
    """A serialization roundtrip yields an equivalent object."""
    assert_roundtrip_consistency(kernel)
