"""Test for initial simple input, recommendation and adding fake results. Fake target
measurements are simulated for each round. Noise is added every second round.
From the three recommendations only one is actually added to test the matching and
metadata. Target objective is minimize to test computational transformation.
"""

import pytest
from baybe.parameters import Parameter
from baybe.utils import add_fake_results, add_parameter_noise


@pytest.mark.parametrize("parameter_names", [["Custom_1", "Custom_2"]])
def test_run_iterations(baybe_one_maximization_target, n_iterations, batch_quantity):
    """
    Test if iterative loop runs with custom parameters.
    """
    baybe_obj = baybe_one_maximization_target

    for _ in range(n_iterations):
        rec = baybe_obj.recommend(batch_quantity=batch_quantity)

        print(rec)
        print(baybe_obj.searchspace.discrete.exp_rep)
        print(baybe_obj.searchspace.discrete.comp_rep)

        add_fake_results(rec, baybe_obj)
        add_parameter_noise(rec, baybe_obj, noise_level=0.1)

        baybe_obj.add_results(rec)

    # This test needs to clear the lru cache, otherwise it causes HashableDict to crash
    Parameter._create.cache_clear()  # pylint: disable=protected-access
