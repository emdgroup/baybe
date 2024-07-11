"""Kernel tests."""

from hypothesis import given

from baybe.kernels.base import Kernel

from .hypothesis_strategies.kernels import kernels


@given(kernels())
def test_kernel_assembly(kernel: Kernel):
    """Turning a BayBE kernel into a GPyTorch kernel raises no errors."""
    kernel.to_gpytorch()
