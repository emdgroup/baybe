# pylint: disable=missing-function-docstring, missing-module-docstring
# TODO: This file needs to be refactored.

from typing import get_args, get_type_hints

import pytest

from baybe.searchspace import SearchSpaceType
from baybe.strategies.bayesian import (
    BayesianRecommender,
    NaiveHybridRecommender,
    SequentialGreedyRecommender,
)
from baybe.strategies.recommender import NonPredictiveRecommender, Recommender
from baybe.surrogate import SurrogateModel
from baybe.utils.basic import get_subclasses
from baybe.utils.dataframe import add_fake_results, add_parameter_noise

########################################################################################
# Settings of the individual components to be tested
########################################################################################
valid_acquisition_functions = get_args(
    get_type_hints(BayesianRecommender.__init__)["acquisition_function_cls"]
)
# TODO: refactor code to avoid the set deduplication below
valid_surrogate_models = list({cls.type for cls in get_subclasses(SurrogateModel)})
valid_initial_recommenders = [cls() for cls in get_subclasses(NonPredictiveRecommender)]
valid_discrete_recommenders = [
    cls()
    for cls in get_subclasses(Recommender)
    if cls.compatibility
    in [SearchSpaceType.DISCRETE, SearchSpaceType.HYBRID, SearchSpaceType.EITHER]
]
valid_continuous_recommenders = [
    cls()
    for cls in get_subclasses(Recommender)
    if cls.compatibility
    in [SearchSpaceType.CONTINUOUS, SearchSpaceType.HYBRID, SearchSpaceType.EITHER]
]
# List of all hybrid recommenders with default attributes. Is extended with other lists
# of hybird recommenders like naive ones or recommenders not using default arguments
valid_hybrid_recommenders = [
    cls()
    for cls in get_subclasses(Recommender)
    if cls.compatibility == SearchSpaceType.HYBRID
]
# List of SequentialGreedy Recommender with different sampling strategies.
sampling_strategies = [
    # Valid combinations
    ("None", 0.0),
    ("None", 1.0),
    ("Farthest", 0.2),
    ("Farthest", 0.5),
    ("Random", 0.2),
    ("Random", 0.5),
]
valid_hybrid_sequential_greedy_recommenders = [
    SequentialGreedyRecommender(hybrid_sampler=sampler, sampling_percentage=per)
    for sampler, per in sampling_strategies
]

valid_discrete_non_predictive_recommenders = [
    cls()
    for cls in get_subclasses(NonPredictiveRecommender)
    if cls.compatibility
    in [SearchSpaceType.DISCRETE, SearchSpaceType.EITHER, SearchSpaceType.HYBRID]
]
valid_discrete_bayesian_recommenders = [
    cls()
    for cls in get_subclasses(BayesianRecommender)
    if cls.compatibility
    in [SearchSpaceType.DISCRETE, SearchSpaceType.EITHER, SearchSpaceType.HYBRID]
]
valid_naive_hybrid_recommenders = [
    NaiveHybridRecommender(
        disc_recommender=disc, cont_recommender=SequentialGreedyRecommender()
    )
    for disc in [
        *valid_discrete_non_predictive_recommenders,
        *valid_discrete_bayesian_recommenders,
    ]
]

valid_hybrid_recommenders.extend(valid_naive_hybrid_recommenders)
valid_hybrid_recommenders.extend(valid_hybrid_sequential_greedy_recommenders)

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

        baybe.add_measurements(rec)


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


@pytest.mark.slow
@pytest.mark.parametrize("recommender", valid_hybrid_recommenders)
@pytest.mark.parametrize(
    "parameter_names",
    [["Categorical_1", "SomeSetting", "Num_disc_1", "Conti_finite1", "Conti_finite2"]],
)
def test_iter_recommender_hybrid(baybe, n_iterations, batch_quantity):
    run_iterations(baybe, n_iterations, batch_quantity)
