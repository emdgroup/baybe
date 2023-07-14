"""Test for the serialization of naive hybrid recommenders."""

import pytest

from baybe.core import BayBE
from baybe.parameters import NumericContinuous, NumericDiscrete
from baybe.searchspace import SearchSpace, SearchSpaceType
from baybe.strategies.bayesian import (
    NaiveHybridRecommender,
    SequentialGreedyRecommender,
)
from baybe.strategies.recommender import BayesianRecommender, NonPredictiveRecommender
from baybe.strategies.strategy import Strategy
from baybe.targets import NumericalTarget, Objective
from baybe.utils import get_subclasses

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


@pytest.mark.parametrize("hybrid_recommender", valid_naive_hybrid_recommenders)
def test_serialization_without_recommendation(hybrid_recommender):
    """Serialize all possible hybrid recommender objects and test for equality"""
    parameters = [
        NumericDiscrete(name="disc", values=[1, 5, 10], tolerance=0.2),
        NumericContinuous(name="cont", bounds=(0, 1)),
    ]
    targets = [NumericalTarget(name="Yield", mode="MAX")]
    baybe_orig = BayBE(
        searchspace=SearchSpace.create(parameters=parameters),
        objective=Objective(mode="SINGLE", targets=targets),
        strategy=Strategy(recommender=hybrid_recommender),
    )
    baybe_orig_string = baybe_orig.to_json()
    baybe_recreate = BayBE.from_json(baybe_orig_string)
    assert baybe_orig == baybe_recreate
