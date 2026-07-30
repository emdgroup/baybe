"""Serialization tests for the LLM recommender."""

from baybe.recommenders import RandomRecommender, TwoPhaseMetaRecommender
from baybe.recommenders.pure.llm.llm import LLMRecommender
from tests.serialization.utils import assert_roundtrip_consistency


def test_llm_recommender_roundtrip():
    """An LLMRecommender survives a serialization roundtrip.

    Pure recommenders do not carry the serialization mix-in themselves; they are
    serialized through an enclosing (meta) recommender, which is what we exercise here.
    Serialization does not touch LiteLLM, so no optional dependencies are required.
    """
    recommender = TwoPhaseMetaRecommender(
        initial_recommender=LLMRecommender(
            model="dummy-provider/dummy-model",
            experiment_description="Optimize a direct arylation reaction.",
            objective_description="Maximize the reaction yield.",
            format_instructions="Return a JSON array only.",
            recovery_model="dummy-provider/dummy-recovery-model",
            litellm_args={"temperature": 0.2},
            recovery_litellm_args={"temperature": 0.0},
        ),
        recommender=RandomRecommender(),
    )
    assert_roundtrip_consistency(recommender)
