"""
Tests for error -free running of some iterations
"""
from abc import ABC
from typing import get_args, get_type_hints

import pytest
from baybe.core import BayBE, BayBEConfig
from baybe.strategy import InitialStrategy, Strategy
from baybe.surrogate import SurrogateModel
from baybe.utils import add_fake_results, add_parameter_noise, subclasses_recursive

# Dictionary containing items describing config tests that should throw an error.
# Key is a string describing the test and is displayed by pytest. Each value is a pair
# of the first item being the config dictionary update that is done to the default
# fixture and the second item being the expected exception type.
config_updates = {
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
    "aq_random": {  # is not covered by the loop below hence added manually
        "strategy": {
            "recommender_cls": "RANDOM",
            "initial_strategy": "RANDOM",
        }
    },
}


# Generate dynamic lists of configurations based on implementation
valid_surrogate_models = [
    cls.type for cls in subclasses_recursive(SurrogateModel) if ABC not in cls.__bases__
]
valid_init_strats = [
    cls.type
    for cls in subclasses_recursive(InitialStrategy)
    if ABC not in cls.__bases__
]
# AQ function type hint looks like this:
# Union[Literal["PM", ...], Type[AcquisitionFunction]]
valid_aq_functions = get_args(
    get_args(get_type_hints(Strategy)["acquisition_function_cls"])[0]
)

for itm in valid_aq_functions:
    # TODO: The recommender class is fixed here to avoid getting invalid combinations of
    #   the default "SEQUENTIAL_GREEDY" class and non-MC acquisition functions.
    #   This selection should be done/checked automatically with root validators at some
    #   point and probably there should be a separate test for such config problems.
    config_updates.update(
        {
            f"aq_{itm}": {
                "strategy": {
                    "acquisition_function_cls": itm,
                    "recommender_cls": "UNRESTRICTED_RANKING",
                }
            }
        }
    )
for itm in valid_surrogate_models:
    config_updates.update(
        {
            f"surrogate_{itm}": {
                "strategy": {
                    "surrogate_model_cls": itm,
                }
            },
        }
    )
for itm in valid_init_strats:
    config_updates.update(
        {
            f"init_{itm}": {
                "strategy": {
                    "initial_strategy": itm,
                }
            },
        }
    )

# List of tests that are expected to fail (still missing implementation etc)
xfails = ["target_multi"]


@pytest.mark.slow
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
