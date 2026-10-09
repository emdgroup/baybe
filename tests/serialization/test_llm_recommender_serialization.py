"""LLM recommender serialization tests."""

import hypothesis.strategies as st
from hypothesis import given

from baybe.recommenders import RandomRecommender
from baybe.recommenders.meta.llm import LLMTwoPhaseRecommender
from baybe.recommenders.pure.llm.llm import LLMRecommender
from tests.conftest import select_recommender
from tests.hypothesis_strategies.recommenders import llm_recommenders
from tests.serialization.utils import roundtrip


@given(data=st.data())
def test_llm_recommender_roundtrip(data: st.DataObject):
    """A serialization roundtrip yields an equivalent object."""
    recommender = data.draw(llm_recommenders())
    roundtripped = LLMRecommender.from_json(recommender.to_json())
    assert roundtripped == recommender


def test_stashed_context_not_serialized():
    """Private stashed fields do not appear in the serialized form."""
    recommender = LLMRecommender(model="test/model", experiment_description="Test.")
    recommender._objective = "should not be serialized"
    recommender._measurements = "should not be serialized"
    recommender._pending_experiments = "should not be serialized"

    serialized = recommender.to_dict()
    assert "_objective" not in serialized
    assert "_measurements" not in serialized
    assert "_pending_experiments" not in serialized

    roundtripped = LLMRecommender.from_dict(serialized)
    assert roundtripped._objective is None
    assert roundtripped._measurements is None
    assert roundtripped._pending_experiments is None


_LLM_KWARGS = {"model": "test/model", "experiment_description": "Test."}


def test_llm_two_phase_state_serialization():
    """The LLM two-phase recommender preserves its switching state."""
    llm_rec = LLMRecommender(**_LLM_KWARGS)
    secondary = RandomRecommender()
    recommender = LLMTwoPhaseRecommender(
        initial_recommender=llm_rec,
        recommender=secondary,
        switch_after=1,
    )

    assert select_recommender(recommender, 0) is llm_rec
    assert select_recommender(recommender, 1) is secondary

    recommender2 = roundtrip(recommender)
    assert recommender2 == recommender
    rec = select_recommender(recommender2, 1)
    assert rec == secondary
