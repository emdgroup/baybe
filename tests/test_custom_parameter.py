"""Test for initial simple input, recommendation and adding fake results.

Fake target measurements are simulated for each round. Noise is added every second
round. From the three recommendations only one is actually added to test the matching
and metadata. Target objective is minimize to test computational transformation.
"""

import pytest

from .conftest import run_iterations


@pytest.mark.parametrize("parameter_names", [["Custom_1", "Custom_2"]])
def test_run_iterations(campaign, n_iterations, batch_size):
    """Test if iterative loop runs with custom parameters."""
    run_iterations(campaign, n_iterations, batch_size)
