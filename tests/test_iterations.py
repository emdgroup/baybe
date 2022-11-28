"""
Tests for error -free running of some iterations
"""

import pytest
from baybe.core import BayBE, BayBEConfig

# Dictionary containing items describing config tests that should throw an error.
# Key is a string describing the test and is displayed by pytest. Each value is a pair
# of the first item being the config dictionary update that is done to the default
# fixture and the second item being the expected exception type.
from baybe.utils import add_fake_results, add_parameter_noise

config_updates = {
    "aq_posterior_mean": {
        "strategy": {
            "acquisition_function_cls": "PM",
        }
    },
    "aq_expected_improvement": {
        "strategy": {
            "acquisition_function_cls": "EI",
        }
    },
    "aq_probability_of_improvement": {
        "strategy": {
            "acquisition_function_cls": "PI",
        }
    },
    "aq_upper_confidence_bound": {
        "strategy": {
            "acquisition_function_cls": "UCB",
        }
    },
    "aq_random": {
        "strategy": {
            "recommender_cls": "RANDOM",
            "initial_strategy": "RANDOM",
        }
    },
    "init_random": {
        "strategy": {
            "initial_strategy": "RANDOM",
        }
    },
    "init_kmedoids": {
        "strategy": {
            "initial_strategy": "PAM",
        }
    },
    "init_kmeans": {
        "strategy": {
            "initial_strategy": "KMEANS",
        }
    },
    "init_gaussian_mixture_model": {
        "strategy": {
            "initial_strategy": "GMM",
        }
    },
    "init_farthest_point_sampling": {
        "strategy": {
            "initial_strategy": "FPS",
        }
    },
    "surrogate_gaussian_process": {
        "strategy": {
            "surrogate_model_cls": "GP",
        }
    },
    "surrogate_mean_prediction": {
        "strategy": {
            "surrogate_model_cls": "MP",
        }
    },
    "surrogate_random_forest": {
        "strategy": {
            "surrogate_model_cls": "RF",
        }
    },
    "surrogate_ngboost": {
        "strategy": {
            "surrogate_model_cls": "NG",
        }
    },
    "surrogate_bayesian_linear": {
        "strategy": {
            "surrogate_model_cls": "BL",
        }
    },
    "target_single_max": {
        "objective": {
            "mode": "SINGLE",
            "targets": [
                {
                    "name": "Target_1",
                    "type": "NUM",
                    "mode": "MAX",
                },
            ],
        }
    },
    "target_single_min": {
        "objective": {
            "mode": "SINGLE",
            "targets": [
                {
                    "name": "Target_1",
                    "type": "NUM",
                    "mode": "MIN",
                },
            ],
        }
    },
    "target_single_match_bell": {
        "objective": {
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
        }
    },
    "target_single_match_triangular": {
        "objective": {
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
        }
    },
    "target_desirability_mean": {
        "objective": {
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
        }
    },
    "target_desirability_geom_mean": {
        "objective": {
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
        }
    },
    "target_multi": {
        "objective": {  # Not Implemented Yet
            "mode": "MULTI",
            "targets": [
                {
                    "name": "Target_1",
                    "type": "NUM",
                    "mode": "MAX",
                },
                {
                    "name": "Target_2",
                    "type": "NUM",
                    "mode": "MIN",
                },
            ],
        }
    },
}

# List of tests that are expected to fail (still missing implementation etc)
xfails = ["target_multi"]


@pytest.mark.parametrize("config_update_key", config_updates.keys())
def test_run_iteration(
    config_basic_1target,
    n_iterations,
    batch_quantity,
    config_update_key,
    good_reference_values,
):
    """
    Test whether the given settings can run some iterations without error.
    """
    if config_update_key in xfails:
        pytest.xfail()

    config_update = config_updates[config_update_key]
    config_basic_1target.update(config_update)

    config = BayBEConfig(**config_basic_1target)
    baybe_obj = BayBE(config)

    for k in range(n_iterations):
        rec = baybe_obj.recommend(batch_quantity=batch_quantity)

        add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
        if k % 2:
            add_parameter_noise(rec, baybe_obj, noise_level=0.1)

        baybe_obj.add_results(rec)
