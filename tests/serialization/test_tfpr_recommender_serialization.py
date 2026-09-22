"""Serialization tests for the TFPR recommender."""

import pytest
from hypothesis import given

from baybe import Campaign
from baybe.objectives import ParetoObjective
from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders.pure.tfpr import TFPRRecommender
from baybe.serialization.core import converter
from baybe.targets import NumericalTarget
from tests.hypothesis_strategies.recommenders import tfpr_recommenders
from tests.serialization.utils import assert_roundtrip_consistency


@given(recommender=tfpr_recommenders())
def test_tfpr_recommender_cattrs_roundtrip(recommender: TFPRRecommender) -> None:
    """A cattrs roundtrip yields an equivalent TFPR recommender."""
    dct = converter.unstructure(recommender)
    roundtrip = converter.structure(dct, TFPRRecommender)
    assert roundtrip == recommender


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(True, 1, id="boolean"),
        pytest.param(1.9, 1, id="fractional"),
    ],
)
def test_tfpr_recommender_structures_weight_inputs(
    value: bool | float, expected: int
) -> None:
    """Cattrs applies the documented integer weight conversion."""
    recommender = converter.structure({"weights": {"target": value}}, TFPRRecommender)

    assert recommender.weights == {"target": expected}


def test_campaign_with_tfpr_recommender_serialization() -> None:
    """A campaign containing a TFPR recommender roundtrips consistently."""
    searchspace = NumericalDiscreteParameter("p", [0, 1]).to_searchspace()
    objective = ParetoObjective([NumericalTarget("t1"), NumericalTarget("t2")])
    campaign = Campaign(
        searchspace,
        objective,
        recommender=TFPRRecommender(
            weights={"t1": 3}, tolerances={"t2": 0.1}, top_fraction=0.5
        ),
    )

    assert_roundtrip_consistency(campaign)
