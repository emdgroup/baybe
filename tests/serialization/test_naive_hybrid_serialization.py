"""Test for the serialization of naive hybrid recommenders."""

import pytest

from baybe.campaign import Campaign
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.naive import NaiveHybridSpaceRecommender
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.recommenders.pure.bayesian.botorch import (
    BotorchRecommender,
)
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpaceType
from baybe.utils.basic import get_subclasses

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
            disc_recommender=disc, cont_recommender=BotorchRecommender()
        )
    )
    for disc in [
        *valid_discrete_non_predictive_recommenders,
        *valid_discrete_bayesian_recommenders,
    ]
]


@pytest.mark.parametrize("recommender", valid_naive_hybrid_recommenders)
@pytest.mark.parametrize(
    "parameter_names",
    [["Categorical_1", "SomeSetting", "Num_disc_1", "Conti_finite1", "Conti_finite2"]],
)
def test_serialization_without_recommendation(campaign):
    """Serialize all possible hybrid recommender objects and test for equality."""
    campaign_orig_string = campaign.to_json()
    campaign_recreate = Campaign.from_json(campaign_orig_string)
    assert campaign == campaign_recreate
