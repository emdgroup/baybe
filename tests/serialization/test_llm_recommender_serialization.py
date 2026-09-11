"""Serialization tests for the LLM recommender."""

from hypothesis import given

from baybe.recommenders.pure.llm.llm import LLMRecommender
from tests.hypothesis_strategies.recommenders import llm_recommenders


@given(llm_recommenders())
def test_llm_recommender_roundtrip(recommender: LLMRecommender):
    """An LLMRecommender survives a serialization roundtrip across all configurations.

    Serialization does not touch LiteLLM, so no optional dependencies are required.
    """
    assert LLMRecommender.from_json(recommender.to_json()) == recommender
