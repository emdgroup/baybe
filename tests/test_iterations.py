"""
Tests for error -free running of some iterations
"""
import random
from abc import ABC
from typing import get_args, get_type_hints

import numpy as np

import pytest
import torch

from baybe.core import BayBE, BayBEConfig
from baybe.recommender import Recommender
from baybe.strategy import Strategy
from baybe.surrogate import SurrogateModel
from baybe.utils import add_fake_results, add_parameter_noise, subclasses_recursive

# Dictionary containing items describing config tests that should throw an error.
# Key is a string describing the test and is displayed by pytest. Each value is a pair
# of the first item being the config dictionary update that is done to the default
# fixture and the second item being the expected exception type.
config_updates_discrete = {
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
            "initial_recommender_cls": "RANDOM",
        }
    },
}
config_updates_continuous = {}
config_updates_hybrid = {}


# Generate dynamic lists of configurations based on implementation
valid_surrogate_models = [
    cls.type for cls in subclasses_recursive(SurrogateModel) if ABC not in cls.__bases__
]
valid_initial_recommenders = [
    subclass_name
    for subclass_name, subclass in Recommender.SUBCLASSES.items()
    if subclass.is_model_free
]
# AQ function type hint looks like this:
# Union[Literal["PM", ...], Type[AcquisitionFunction]]
valid_aq_functions = get_args(
    get_args(get_type_hints(Strategy)["acquisition_function_cls"])[0]
)
valid_purely_discrete_recommenders = [
    name
    for name, subclass in Recommender.SUBCLASSES.items()
    if (subclass.compatible_discrete and not subclass.compatible_continuous)
]
valid_purely_continuous_recommenders = [
    name
    for name, subclass in Recommender.SUBCLASSES.items()
    if (not subclass.compatible_discrete and subclass.compatible_continuous)
]
valid_hybrid_recommenders = [
    name
    for name, subclass in Recommender.SUBCLASSES.items()
    if (subclass.compatible_discrete and subclass.compatible_continuous)
]

for itm in valid_aq_functions:
    # TODO: The recommender class is fixed here to avoid getting invalid combinations of
    #   the default "SEQUENTIAL_GREEDY_DISCRETE" class and non-MC acquisition functions.
    #   This selection should be done/checked automatically with root validators at some
    #   point and probably there should be a separate test for such config problems.
    config_updates_discrete.update(
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
    config_updates_discrete.update(
        {
            f"surrogate_{itm}": {
                "strategy": {
                    "surrogate_model_cls": itm,
                }
            },
        }
    )
for itm in valid_initial_recommenders:
    config_updates_discrete.update(
        {
            f"init_{itm}": {
                "strategy": {
                    "initial_recommender_cls": itm,
                }
            },
        }
    )
for itm in valid_purely_discrete_recommenders:
    config_updates_discrete.update(
        {
            f"rec_{itm}": {
                "strategy": {
                    "recommender_cls": itm,
                }
            }
        }
    )
for itm in valid_purely_continuous_recommenders:
    config_updates_continuous.update(
        {
            f"rec_{itm}": {
                "strategy": {
                    "recommender_cls": itm,
                }
            }
        }
    )


# List of tests that are expected to fail (still missing implementation etc)
xfails = ["target_multi"]


@pytest.mark.slow
@pytest.mark.parametrize("config_update_key", config_updates_discrete.keys())
def test_run_iteration_discrete(
    config_discrete_1target,
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

    config_update = config_updates_discrete[config_update_key]
    config_discrete_1target.update(config_update)

    config = BayBEConfig(**config_discrete_1target)
    baybe_obj = BayBE(config)

    for k in range(n_iterations):
        rec = baybe_obj.recommend(batch_quantity=batch_quantity)

        add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
        if k % 2:
            add_parameter_noise(rec, baybe_obj, noise_level=0.1)

        baybe_obj.add_results(rec)


@pytest.mark.slow
@pytest.mark.parametrize("config_update_key", config_updates_continuous.keys())
def test_run_iteration_continuous(
    config_continuous_1target,
    n_iterations,
    batch_quantity,
    config_update_key,
):
    """
    Test whether the given settings can run some iterations without error.
    """
    if config_update_key in xfails:
        pytest.xfail()

    config_update = config_updates_continuous[config_update_key]
    config_continuous_1target.update(config_update)

    config = BayBEConfig(**config_continuous_1target)
    baybe_obj = BayBE(config)

    for k in range(n_iterations):
        rec = baybe_obj.recommend(batch_quantity=batch_quantity)

        add_fake_results(rec, baybe_obj)
        if k % 2:
            add_parameter_noise(rec, baybe_obj, noise_level=0.1)

        baybe_obj.add_results(rec)


def test_recommendation_caching(
    config_discrete_1target,
    batch_quantity,
    good_reference_values,
):
    """
    Test recommendation caching and consistency
    """
    config = BayBEConfig(**config_discrete_1target)
    baybe_obj = BayBE(config)

    # Add results of an inital recommendation
    rec0 = baybe_obj.recommend(batch_quantity=batch_quantity)
    add_fake_results(rec0, baybe_obj, good_reference_values=good_reference_values)
    add_parameter_noise(rec0, baybe_obj, noise_level=0.1)
    baybe_obj.add_results(rec0)

    # Perform iterations of calling recommend without adding data
    rec1 = baybe_obj.recommend(batch_quantity=batch_quantity)
    for k in range(3):
        rec = baybe_obj.recommend(batch_quantity=batch_quantity)

        assert rec.equals(rec1)

        # Change random seed to increase likelihood of differences in the model fit
        # (which should not be performed)
        torch.manual_seed(1337 + k)
        random.seed(1337 + k)
        np.random.seed(1337 + k)

    # Add fake results
    rec = baybe_obj.recommend(batch_quantity=batch_quantity)
    add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
    add_parameter_noise(rec0, baybe_obj, noise_level=0.1)
    baybe_obj.add_results(rec)

    # Check that recommendations now are different
    rec = baybe_obj.recommend(batch_quantity=batch_quantity)
    assert not rec.equals(rec1)
