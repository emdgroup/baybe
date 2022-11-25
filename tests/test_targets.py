"""
Tests for basic input-output nad iterative loop.
"""
import pytest

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise

dict_objective_variants = {
    "single_max": {
        "mode": "SINGLE",
        "targets": [
            {
                "name": "Target_1",
                "type": "NUM",
                "mode": "MAX",
                # "bounds": (0, 100),
                # "bounds_transform_func": "BELL",
            },
        ],
    },
    "single_min": {
        "mode": "SINGLE",
        "targets": [
            {
                "name": "Target_1",
                "type": "NUM",
                "mode": "MIN",
                # "bounds": (0, 100),
                # "bounds_transform_func": "BELL",
            },
        ],
    },
    "single_match_bell": {
        "mode": "SINGLE",
        "targets": [
            {
                "name": "Target_1",
                "type": "NUM",
                "mode": "MATCH",
                "bounds": (0, 100),
                "bounds_transform_func": "BELL",
            },
        ],
    },
    "single_match_triangular": {
        "mode": "SINGLE",
        "targets": [
            {
                "name": "Target_1",
                "type": "NUM",
                "mode": "MATCH",
                "bounds": (0, 100),
                "bounds_transform_func": "TRIANGULAR",
            },
        ],
    },
    "desirability_mean": {
        "mode": "DESIRABILITY",
        "combine_func": "MEAN",
        "targets": [
            {
                "name": "Target_1",
                "type": "NUM",
                "mode": "MAX",
                "bounds": (0, 100),
            },
            {
                "name": "Target_2",
                "type": "NUM",
                "mode": "MIN",
                "bounds": (0, 100),
            },
            {
                "name": "Target_3",
                "type": "NUM",
                "mode": "MATCH",
                "bounds": [45, 55],
            },
        ],
    },
    "desirability_geom_mean": {
        "mode": "DESIRABILITY",
        "combine_func": "GEOM_MEAN",
        "targets": [
            {
                "name": "Target_1",
                "type": "NUM",
                "mode": "MAX",
                "bounds": (0, 100),
            },
            {
                "name": "Target_2",
                "type": "NUM",
                "mode": "MIN",
                "bounds": (0, 100),
            },
            {
                "name": "Target_3",
                "type": "NUM",
                "mode": "MATCH",
                "bounds": [45, 55],
            },
        ],
    },
    # "multi": { # Not Implemented Yet
    #     "mode": "MULTI",
    #     "targets": [
    #         {
    #             "name": "Target_1",
    #             "type": "NUM",
    #             "mode": "MAX",
    #         },
    #         {
    #             "name": "Target_2",
    #             "type": "NUM",
    #             "mode": "MIN",
    #         },
    #     ],
    # },
}


@pytest.mark.parametrize("config_update_key", dict_objective_variants.keys())
def test_run_iterations(
    config_basic_1target,
    n_iterations,
    good_reference_values,
    batch_quantity,
    config_update_key,
):
    """
    Test running some iterations with fake results and basic parameters.
    """
    config_basic_1target["objective"] = dict_objective_variants[config_update_key]
    config = BayBEConfig(**config_basic_1target)
    baybe_obj = BayBE(config)

    for k in range(n_iterations):
        rec = baybe_obj.recommend(batch_quantity=batch_quantity)

        add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
        if k % 2:
            add_parameter_noise(rec, baybe_obj, noise_level=0.1)

        baybe_obj.add_results(rec)
