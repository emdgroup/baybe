"""Test for initial simple input, recommendation and adding fake results. Fake target
measurements are simulated for each round. Noise is added every second round.
From the three recommendations only one is actually added to test the matching and
metadata. Target objective is minimize to test computational transformation.
"""
import pandas as pd
import pytest

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise


@pytest.mark.xfail
def test_run_iterations(config_basic_1target, n_iterations, good_reference_values):
    """
    Test if iterative loop runs with custom parameters.
    """
    custom_df = pd.DataFrame(
        {
            "D1": [1.1, 1.4, 1.7],
            "D2": [11, 23, 55],
            "D3": [-4, -13, 4],
        },
        index=["mol1", "mol2", "mol3"],
    )
    custom_df2 = pd.DataFrame(
        {
            "desc1": [1.1, 1.4, 1.7],
            "desc2": [55, 23, 3],
            "desc3": [4, 5, 6],
        },
        index=["A", "B", "C"],
    )

    config_basic_1target["parameters"].append(
        {
            "name": "Custom_1",
            "type": "CUSTOM",
            "data": custom_df,
        }
    )
    config_basic_1target["parameters"].append(
        {
            "name": "Custom_2",
            "type": "CUSTOM",
            "data": custom_df2,
        }
    )
    config = BayBEConfig(**config_basic_1target)
    baybe_obj = BayBE(config)

    for _ in range(n_iterations):
        rec = baybe_obj.recommend(batch_quantity=3)

        add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
        add_parameter_noise(rec, baybe_obj, noise_level=0.1)

        baybe_obj.add_results(rec)
