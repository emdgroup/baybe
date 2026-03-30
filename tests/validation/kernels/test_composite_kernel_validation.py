"""Validation tests for composite kernels."""

import pytest
from pytest import param

from baybe.kernels.basic import MaternKernel
from baybe.kernels.composite import AdditiveKernel, ProductKernel, ScaleKernel

base_kernel = MaternKernel()


@pytest.mark.parametrize(
    ("base_kernel", "outputscale_prior", "outputscale_initial_value", "error"),
    [
        param("wrong", None, None, TypeError, id="not_a_kernel"),
        param(base_kernel, "wrong", None, TypeError, id="not_a_prior"),
        param(base_kernel, None, "wrong", ValueError, id="not_a_scale"),
        param(base_kernel, None, -1, ValueError, id="negative_scale"),
        param(base_kernel, None, 0, ValueError, id="zero_scale"),
    ],
)
def test_scale_kernel_validation(
    base_kernel, outputscale_prior, outputscale_initial_value, error
):
    """Providing invalid kernel arguments raises an exception."""
    with pytest.raises(error):
        ScaleKernel(base_kernel, outputscale_prior, outputscale_initial_value)


@pytest.mark.parametrize("kernel_cls", [AdditiveKernel, ProductKernel])
@pytest.mark.parametrize(
    ("base_kernels", "error"),
    [
        param(1, TypeError, id="not_an_iterable"),
        param([], ValueError, id="empty_iterable"),
        param([base_kernel], ValueError, id="only_one_kernel"),
        param([1, base_kernel], TypeError, id="not_a_kernel"),
    ],
)
def test_additive_and_product_kernel_validation(kernel_cls, base_kernels, error):
    """Providing invalid kernel arguments raises an exception."""
    with pytest.raises(error):
        kernel_cls(base_kernels)
