"""TFPR recommender serialization tests."""

import hypothesis.strategies as st
import pytest
from hypothesis import given
from pytest import param

from baybe.recommenders import TFPRRecommender, TwoPhaseMetaRecommender
from baybe.serialization import converter
from tests.hypothesis_strategies.recommenders import tfpr_recommenders
from tests.serialization.utils import assert_roundtrip_consistency


@given(data=st.data())
def test_tfpr_recommender_roundtrip(data: st.DataObject):
    """A serialization roundtrip yields an equivalent object."""
    recommender = data.draw(tfpr_recommenders())
    dct = converter.unstructure(recommender)
    assert converter.structure(dct, TFPRRecommender) == recommender


@pytest.mark.parametrize(
    ("weights", "tolerances"),
    [
        param(None, None, id="defaults"),
        param({"t1": 2.5, "t2": 0.0}, {"t1": 0.1, "t2": 0.0}, id="custom"),
    ],
)
def test_private_ranking_settings_survive_nested_roundtrip(weights, tolerances):
    """The private ranking settings survive a roundtrip inside a meta recommender."""
    recommender = TFPRRecommender()
    recommender._weights = weights
    recommender._tolerances = tolerances
    assert_roundtrip_consistency(TwoPhaseMetaRecommender(recommender=recommender))


def test_transient_objective_not_serialized():
    """The encountered objective is not part of the serialized form."""
    recommender = TFPRRecommender()
    recommender._objective = "should not be serialized"

    dct = converter.unstructure(recommender)
    assert "_objective" not in dct
    assert converter.structure(dct, TFPRRecommender)._objective is None


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        param("_weights", {"t1": -1.0}, ">= 0", id="negative_weight"),
        param("_weights", {"t1": 0.0}, "positive", id="zero_weights"),
        param("_tolerances", {"t1": float("inf")}, "inf", id="infinite_tolerance"),
    ],
)
def test_private_ranking_settings_validated_on_deserialization(key, value, match):
    """Invalid private ranking settings are rejected during deserialization."""
    dct = converter.unstructure(TFPRRecommender()) | {key: value}
    with pytest.raises(ValueError, match=match):
        converter.structure(dct, TFPRRecommender)
