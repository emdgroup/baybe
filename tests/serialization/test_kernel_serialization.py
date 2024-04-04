"""Test serialization of kernels."""

from hypothesis import given

from baybe.kernels import MaternKernel
from tests.hypothesis_strategies.kernels import matern_kernels


@given(matern_kernels)
def test_matern_kernel_roundtrip(kernel: MaternKernel):
    string = kernel.to_json()
    kernel2 = MaternKernel.from_json(string)
    assert kernel == kernel2, (kernel, kernel2)
