"""Validation tests for basic kernels."""

import pytest
from pytest import param

from baybe.kernels.basic import (
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    PiecewisePolynomialKernel,
    PolynomialKernel,
    RBFKernel,
    RFFKernel,
    RQKernel,
)


@pytest.mark.parametrize(
    ("variance_prior", "variance_initial_value", "error"),
    [
        param("wrong", None, TypeError, id="not_a_prior"),
        param(None, "wrong", ValueError, id="not_a_variance"),
        param(None, -1, ValueError, id="negative_variance"),
        param(None, 0, ValueError, id="zero_variance"),
    ],
)
def test_linear_kernel_validation(variance_prior, variance_initial_value, error):
    """Providing invalid kernel arguments raises an exception."""
    with pytest.raises(error):
        LinearKernel(variance_prior, variance_initial_value)


@pytest.mark.parametrize(
    ("nu", "lengthscale_prior", "lengthscale_initial_value", "error"),
    [
        param(0, None, None, ValueError, id="invalid_nu"),
        param(2.5, "wrong", None, TypeError, id="not_a_prior"),
        param(2.5, None, "wrong", ValueError, id="not_a_lengthscale"),
        param(2.5, None, -1, ValueError, id="negative_lengthscale"),
        param(2.5, None, 0, ValueError, id="zero_lengthscale"),
    ],
)
def test_matern_kernel_validation(
    nu, lengthscale_prior, lengthscale_initial_value, error
):
    """Providing invalid kernel arguments raises an exception."""
    with pytest.raises(error):
        MaternKernel(nu, lengthscale_prior, lengthscale_initial_value)


@pytest.mark.parametrize(
    (
        "lengthscale_prior",
        "lengthscale_initial_value",
        "period_length_prior",
        "period_length_initial_value",
        "error",
    ),
    [
        param("wrong", None, None, None, TypeError, id="not_a_lengthscale_prior"),
        param(None, "wrong", None, None, ValueError, id="not_a_lengthscale"),
        param(None, -1, None, None, ValueError, id="negative_lengthscale"),
        param(None, 0, None, None, ValueError, id="zero_lengthscale"),
        param(None, None, "wrong", None, TypeError, id="not_a_period_length_prior"),
        param(None, None, None, "wrong", ValueError, id="not_a_period_length"),
        param(None, None, None, -1, ValueError, id="negative_period_length"),
        param(None, None, None, 0, ValueError, id="zero_period_length"),
    ],
)
def test_periodic_kernel_validation(
    lengthscale_prior,
    lengthscale_initial_value,
    period_length_prior,
    period_length_initial_value,
    error,
):
    """Providing invalid kernel arguments raises an exception."""
    with pytest.raises(error):
        PeriodicKernel(
            lengthscale_prior,
            lengthscale_initial_value,
            period_length_prior,
            period_length_initial_value,
        )


@pytest.mark.parametrize(
    ("q", "lengthscale_prior", "lengthscale_initial_value", "error"),
    [
        param(4, None, None, ValueError, id="invalid_q"),
        param(0, "wrong", None, TypeError, id="not_a_prior"),
        param(0, None, "wrong", ValueError, id="not_a_lengthscale"),
        param(0, None, -1, ValueError, id="negative_lengthscale"),
        param(0, None, 0, ValueError, id="zero_lengthscale"),
    ],
)
def test_piecewise_polynomial_kernel_validation(
    q, lengthscale_prior, lengthscale_initial_value, error
):
    """Providing invalid kernel arguments raises an exception."""
    with pytest.raises(error):
        PiecewisePolynomialKernel(q, lengthscale_prior, lengthscale_initial_value)


@pytest.mark.parametrize(
    ("power", "offset_prior", "offset_initial_value", "error"),
    [
        param(-1, None, None, ValueError, id="invalid_power"),
        param(0, "wrong", None, TypeError, id="not_a_prior"),
        param(0, None, "wrong", ValueError, id="not_a_lengthscale"),
        param(0, None, -1, ValueError, id="negative_lengthscale"),
        param(0, None, 0, ValueError, id="zero_lengthscale"),
    ],
)
def test_polynomial_kernel_validation(power, offset_prior, offset_initial_value, error):
    """Providing invalid kernel arguments raises an exception."""
    with pytest.raises(error):
        PolynomialKernel(power, offset_prior, offset_initial_value)


@pytest.mark.parametrize(
    ("lengthscale_prior", "lengthscale_initial_value", "error"),
    [
        param("wrong", None, TypeError, id="not_a_prior"),
        param(None, "wrong", ValueError, id="not_a_lengthscale"),
        param(None, -1, ValueError, id="negative_lengthscale"),
        param(None, 0, ValueError, id="zero_lengthscale"),
    ],
)
def test_rbf_kernel_validation(lengthscale_prior, lengthscale_initial_value, error):
    """Providing invalid kernel arguments raises an exception."""
    with pytest.raises(error):
        RBFKernel(lengthscale_prior, lengthscale_initial_value)


@pytest.mark.parametrize(
    ("num_samples", "lengthscale_prior", "lengthscale_initial_value", "error"),
    [
        param(0, None, None, ValueError, id="invalid_num_samples"),
        param(1, "wrong", None, TypeError, id="not_a_prior"),
        param(1, None, "wrong", ValueError, id="not_a_lengthscale"),
        param(1, None, -1, ValueError, id="negative_lengthscale"),
        param(1, None, 0, ValueError, id="zero_lengthscale"),
    ],
)
def test_rff_kernel_validation(
    num_samples, lengthscale_prior, lengthscale_initial_value, error
):
    """Providing invalid kernel arguments raises an exception."""
    with pytest.raises(error):
        RFFKernel(num_samples, lengthscale_prior, lengthscale_initial_value)


@pytest.mark.parametrize(
    ("lengthscale_prior", "lengthscale_initial_value", "error"),
    [
        param("wrong", None, TypeError, id="not_a_prior"),
        param(None, "wrong", ValueError, id="not_a_lengthscale"),
        param(None, -1, ValueError, id="negative_lengthscale"),
        param(None, 0, ValueError, id="zero_lengthscale"),
    ],
)
def test_rq_kernel_validation(lengthscale_prior, lengthscale_initial_value, error):
    """Providing invalid kernel arguments raises an exception."""
    with pytest.raises(error):
        RQKernel(lengthscale_prior, lengthscale_initial_value)
