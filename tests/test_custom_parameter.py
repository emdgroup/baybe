"""Tests for the custom parameter."""

import pytest

from .conftest import run_iterations


@pytest.mark.parametrize("parameter_names", [["Custom_1", "Custom_2"]])
def test_run_iterations(campaign, n_iterations, batch_size):
    """Test if iterative loop runs with custom parameters."""
    run_iterations(campaign, n_iterations, batch_size)
