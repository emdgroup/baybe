"""Serialization tests for the LLM recommender."""

from hypothesis import given

from baybe.recommenders import RandomRecommender, TwoPhaseMetaRecommender
from baybe.recommenders.pure.llm.llm import LLMRecommender
from tests.hypothesis_strategies.recommenders import llm_recommenders
from tests.serialization.utils import assert_roundtrip_consistency


@given(llm_recommenders())
def test_llm_recommender_roundtrip(recommender: LLMRecommender):
    """An LLMRecommender survives a serialization roundtrip across all configurations.

    Pure recommenders do not carry the serialization mix-in themselves; they are
    serialized through an enclosing meta recommender, which is what we exercise here.
    Serialization does not touch LiteLLM, so no optional dependencies are required.
    """
    wrapped = TwoPhaseMetaRecommender(
        initial_recommender=recommender,
        recommender=RandomRecommender(),
    )
    assert_roundtrip_consistency(wrapped)
