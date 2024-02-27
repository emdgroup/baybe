# TODO: This file needs to be refactored.
"""Tests various configurations for a small number of iterations."""

from typing import get_args, get_type_hints

import pytest

from baybe.recommenders.meta.base import MetaRecommender
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.naive import NaiveHybridSpaceRecommender
from baybe.recommenders.pure.base import PureRecommender
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.recommenders.pure.bayesian.sequential_greedy import (
    SequentialGreedyRecommender,
)
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpaceType
from baybe.surrogates import get_available_surrogates
from baybe.utils.basic import get_subclasses

from .conftest import run_iterations

########################################################################################
# Settings of the individual components to be tested
########################################################################################
valid_acquisition_functions = get_args(
    get_type_hints(BayesianRecommender.__init__)["acquisition_function_cls"]
)
valid_surrogate_models = [cls() for cls in get_available_surrogates()]
valid_initial_recommenders = [cls() for cls in get_subclasses(NonPredictiveRecommender)]
# TODO the TwoPhaseMetaRecommender below can be removed if the SeqGreedy recommender
#  allows no training data
valid_discrete_recommenders = [
    TwoPhaseMetaRecommender(recommender=cls())
    for cls in get_subclasses(PureRecommender)
    if cls.compatibility
    in [SearchSpaceType.DISCRETE, SearchSpaceType.HYBRID, SearchSpaceType.EITHER]
]
# TODO the TwoPhaseMetaRecommender below can be removed if the SeqGreedy recommender
#  allows no training data
valid_continuous_recommenders = [
    TwoPhaseMetaRecommender(recommender=cls())
    for cls in get_subclasses(PureRecommender)
    if cls.compatibility
    in [SearchSpaceType.CONTINUOUS, SearchSpaceType.HYBRID, SearchSpaceType.EITHER]
]
# List of all hybrid recommenders with default attributes. Is extended with other lists
# of hybird recommenders like naive ones or recommenders not using default arguments
# TODO the TwoPhaseMetaRecommender below can be removed if the SeqGreedy recommender
#  allows no training data
valid_hybrid_recommenders = [
    TwoPhaseMetaRecommender(recommender=cls())
    for cls in get_subclasses(PureRecommender)
    if cls.compatibility == SearchSpaceType.HYBRID
]
# List of SequentialGreedy recommenders with different sampling strategies.
sampling_strategies = [
    # Valid combinations
    ("None", 0.0),
    ("None", 1.0),
    ("Farthest", 0.2),
    ("Farthest", 0.5),
    ("Random", 0.2),
    ("Random", 0.5),
]
# TODO the TwoPhaseMetaRecommender below can be removed if the SeqGreedy recommender
#  allows no training data
valid_hybrid_sequential_greedy_recommenders = [
    TwoPhaseMetaRecommender(
        recommender=SequentialGreedyRecommender(
            hybrid_sampler=sampler, sampling_percentage=per
        )
    )
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
# TODO the TwoPhaseMetaRecommender below can be removed if the SeqGreedy recommender
#  allows no training data
valid_naive_hybrid_recommenders = [
    TwoPhaseMetaRecommender(
        recommender=NaiveHybridSpaceRecommender(
            disc_recommender=disc, cont_recommender=SequentialGreedyRecommender()
        )
    )
    for disc in [
        *valid_discrete_non_predictive_recommenders,
        *valid_discrete_bayesian_recommenders,
    ]
]

valid_hybrid_recommenders.extend(valid_naive_hybrid_recommenders)
valid_hybrid_recommenders.extend(valid_hybrid_sequential_greedy_recommenders)

valid_meta_recommenders = get_subclasses(MetaRecommender)

test_targets = [
    ["Target_max"],
    ["Target_min"],
    ["Target_match_bell"],
    ["Target_match_triangular"],
    ["Target_max_bounded", "Target_min_bounded"],
]


# TODO: The following tests are deactivated because there is currently no Bayesian
#   recommender that can handle non-MC acquisition functions. Once the
#   MarginalRecommender (or similar) is re-added, the acqf-tests can be reactivated.
# @pytest.mark.slow
# @pytest.mark.parametrize("acquisition_function_cls", valid_acquisition_functions)
# def test_iter_acquisition_function(campaign, n_iterations, batch_size):
#     run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize("surrogate_model", valid_surrogate_models)
def test_iter_surrogate_model(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize("recommender", valid_initial_recommenders)
def test_iter_initial_recommender(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize("target_names", test_targets)
def test_iter_targets(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize("recommender", valid_discrete_recommenders)
def test_iter_recommender_discrete(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize("recommender", valid_continuous_recommenders)
@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
def test_iter_recommender_continuous(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.slow
@pytest.mark.parametrize("recommender", valid_hybrid_recommenders)
@pytest.mark.parametrize(
    "parameter_names",
    [["Categorical_1", "SomeSetting", "Num_disc_1", "Conti_finite1", "Conti_finite2"]],
)
def test_iter_recommender_hybrid(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.parametrize("recommender", valid_meta_recommenders, indirect=True)
def test_meta_recommenders(campaign, n_iterations, batch_size):
    run_iterations(campaign, n_iterations, batch_size)
