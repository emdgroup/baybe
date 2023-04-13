# pylint: disable=missing-function-docstring, missing-module-docstring
# TODO: This file needs to be refactored.

from abc import ABC
from typing import get_args, get_type_hints

import pytest

from baybe.searchspace import SearchSpaceType
from baybe.strategies.recommender import (
    BayesianRecommender,
    NonPredictiveRecommender,
    Recommender,
)
from baybe.surrogate import SurrogateModel
from baybe.utils import add_fake_results, add_parameter_noise, subclasses_recursive

########################################################################################
# Settings of the individual components to be tested
########################################################################################
valid_acquisition_functions = get_args(
    get_type_hints(BayesianRecommender.__init__)["acquisition_function_cls"]
)
valid_surrogate_models = list(
    {
        cls.type
        for cls in subclasses_recursive(SurrogateModel)
        if ABC not in cls.__bases__
    }
)
valid_initial_recommenders = [
    cls()
    for cls in subclasses_recursive(NonPredictiveRecommender)
    if ABC not in cls.__bases__
]
valid_discrete_recommenders = [
    cls()
    for cls in Recommender.SUBCLASSES.values()
    if cls.compatibility
    in [SearchSpaceType.DISCRETE, SearchSpaceType.HYBRID, SearchSpaceType.EITHER]
]
valid_continuous_recommenders = [
    cls()
    for cls in Recommender.SUBCLASSES.values()
    if cls.compatibility
    in [SearchSpaceType.CONTINUOUS, SearchSpaceType.HYBRID, SearchSpaceType.EITHER]
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


# TODO: The following tests are deactivated because there is currently no Bayesian
#   recommender that can handle non-MC acquisition functions. Once the
#   MarginalRecommender (or similar) is re-added, the acqf-tests can be reactivated.
# @pytest.mark.slow
# @pytest.mark.parametrize("acquisition_function_cls", valid_acquisition_functions)
# def test_iter_acquisition_function(baybe, n_iterations, batch_quantity):
#     run_iterations(baybe, n_iterations, batch_quantity)


@pytest.mark.slow
@pytest.mark.parametrize("surrogate_model_cls", valid_surrogate_models)
def test_iter_surrogate_model(baybe, n_iterations, batch_quantity):
    run_iterations(baybe, n_iterations, batch_quantity)


@pytest.mark.slow
@pytest.mark.parametrize("initial_recommender", valid_initial_recommenders)
def test_iter_initial_recommender(baybe, n_iterations, batch_quantity):
    run_iterations(baybe, n_iterations, batch_quantity)


@pytest.mark.slow
@pytest.mark.parametrize("target_names", test_targets)
def test_iter_targets(baybe, n_iterations, batch_quantity):
    run_iterations(baybe, n_iterations, batch_quantity)


@pytest.mark.slow
@pytest.mark.parametrize("recommender", valid_discrete_recommenders)
def test_iter_recommender_discrete(baybe, n_iterations, batch_quantity):
    run_iterations(baybe, n_iterations, batch_quantity)


@pytest.mark.slow
@pytest.mark.parametrize("recommender", valid_continuous_recommenders)
@pytest.mark.parametrize("parameter_names", ["Conti_finite1", "Conti_finite2"])
def test_iter_recommender_continuous(baybe, n_iterations, batch_quantity):
    run_iterations(baybe, n_iterations, batch_quantity)
