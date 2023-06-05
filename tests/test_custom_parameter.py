"""Test for initial simple input, recommendation and adding fake results. Fake target
measurements are simulated for each round. Noise is added every second round.
From the three recommendations only one is actually added to test the matching and
metadata. Target objective is minimize to test computational transformation.
"""

import pytest

from baybe.utils import add_fake_results, add_parameter_noise


@pytest.mark.parametrize("parameter_names", [["Custom_1", "Custom_2"]])
def test_run_iterations(baybe, n_iterations, batch_quantity):
    """
    Test if iterative loop runs with custom parameters.
    """
    for _ in range(n_iterations):
        rec = baybe.recommend(batch_quantity=batch_quantity)

        print(rec)
        print(baybe.searchspace.discrete.exp_rep)
        print(baybe.searchspace.discrete.comp_rep)

        add_fake_results(rec, baybe)
        add_parameter_noise(rec, baybe, noise_level=0.1)

        baybe.add_measurements(rec)
