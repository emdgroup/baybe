# pylint: disable=missing-function-docstring, missing-module-docstring
# TODO: This file needs to be refactored.

from abc import ABC
from typing import get_args, get_type_hints

import pytest

from baybe.recommender import Recommender
from baybe.strategy import Strategy
from baybe.surrogate import SurrogateModel
from baybe.utils import add_fake_results, add_parameter_noise, subclasses_recursive

########################################################################################
# Settings of the individual components to be tested
########################################################################################
valid_acquisition_functions = get_args(
    get_type_hints(Strategy)["acquisition_function_cls"]
)
valid_surrogate_models = list(
    {
        cls.type
        for cls in subclasses_recursive(SurrogateModel)
        if ABC not in cls.__bases__
    }
)
valid_initial_recommenders = [
    subclass_name
    for subclass_name, subclass in Recommender.SUBCLASSES.items()
    if subclass.is_model_free
]
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
test_targets = [
    "Target_max",
    "Target_min",
    "Target_match_bell",
    "Target_match_triangular",
    ["Target_max_bounded", "Target_min_bounded"],
]


########################################################################################
# Create tests for each of the above defined settings
# TODO: The following is boilerplate code to avoid the Cartesian product that pytest
#   would create when stacking all parametrizations on a single test function. There
#   must be a better way ...
########################################################################################


def run_iterations(baybe, n_iterations, batch_quantity):
    for k in range(n_iterations):
        rec = baybe.recommend(batch_quantity=batch_quantity)

    add_fake_results(rec, baybe)
    if k % 2:
        add_parameter_noise(rec, baybe, noise_level=0.1)

    baybe.add_results(rec)


@pytest.mark.slow
@pytest.mark.parametrize("acquisition_function_cls", valid_acquisition_functions)
def test_iter_acquisition_function(baybe, n_iterations, batch_quantity):
    run_iterations(baybe, n_iterations, batch_quantity)


@pytest.mark.slow
@pytest.mark.parametrize("surrogate_model_cls", valid_surrogate_models)
def test_iter_surrogate_model(baybe, n_iterations, batch_quantity):
    run_iterations(baybe, n_iterations, batch_quantity)


@pytest.mark.slow
@pytest.mark.parametrize("initial_recommender_cls", valid_initial_recommenders)
def test_iter_initial_recommender(baybe, n_iterations, batch_quantity):
    run_iterations(baybe, n_iterations, batch_quantity)


@pytest.mark.slow
@pytest.mark.parametrize("target_names", test_targets)
def test_iter_targets(baybe, n_iterations, batch_quantity):
    run_iterations(baybe, n_iterations, batch_quantity)


@pytest.mark.slow
@pytest.mark.parametrize("recommender_cls", valid_purely_discrete_recommenders)
def test_iter_recommender_discrete(baybe, n_iterations, batch_quantity):
    run_iterations(baybe, n_iterations, batch_quantity)


@pytest.mark.slow
@pytest.mark.parametrize("recommender_cls", valid_purely_continuous_recommenders)
@pytest.mark.parametrize("parameter_names", ["Conti_finite1", "Conti_finite2"])
def test_iter_recommender_continuous(baybe, n_iterations, batch_quantity):
    run_iterations(baybe, n_iterations, batch_quantity)
