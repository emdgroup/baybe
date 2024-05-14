"""Test serialization of kernels."""

from hypothesis import given

from baybe.kernels.base import Kernel
from tests.hypothesis_strategies.kernels import kernels


@given(kernels())
def test_kernel_roundtrip(kernel: Kernel):
    string = kernel.to_json()
    kernel2 = Kernel.from_json(string)
    assert kernel == kernel2, (kernel, kernel2)
