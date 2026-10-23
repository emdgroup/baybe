"""LLM recommender serialization tests."""

from contextlib import nullcontext

import hypothesis.strategies as st
import pytest
from attrs import evolve
from hypothesis import given
from pytest import param

from baybe.recommenders import RandomRecommender, TwoPhaseMetaRecommender
from baybe.recommenders.pure.llm.llm import LLMRecommender
from tests.conftest import select_recommender
from tests.hypothesis_strategies.recommenders import llm_recommenders
from tests.serialization.utils import roundtrip

_LLM_KWARGS = {"model": "test/model", "experiment_description": "Test."}
_NOT_SERIALIZED = "'litellm_args' of 'LLMRecommender' .* are not serialized"


@given(data=st.data())
def test_llm_recommender_roundtrip(data: st.DataObject):
    """A serialization roundtrip yields an equivalent object without LiteLLM args."""
    recommender = data.draw(llm_recommenders())
    with (
        pytest.warns(UserWarning, match=_NOT_SERIALIZED)
        if recommender.litellm_args
        else nullcontext()
    ):
        string = recommender.to_json()
    roundtripped = LLMRecommender.from_json(string)
    assert roundtripped == evolve(recommender, litellm_args={})


@pytest.mark.parametrize(
    "wrap",
    [
        param(lambda r: r, id="plain"),
        param(lambda r: TwoPhaseMetaRecommender(recommender=r), id="nested"),
    ],
)
def test_litellm_args_not_serialized(wrap):
    """LiteLLM args are dropped from the serialized form, also when nested."""
    recommender = LLMRecommender(**_LLM_KWARGS, litellm_args={"api_key": "secret"})
    with pytest.warns(UserWarning, match=_NOT_SERIALIZED):
        string = wrap(recommender).to_json()
    assert "api_key" not in string
    assert "secret" not in string


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


def test_llm_two_phase_state_serialization():
    """The two-phase recommender with LLM warm start preserves its switching state."""
    secondary = RandomRecommender()
    recommender = TwoPhaseMetaRecommender.with_llm_initial_recommender(
        **_LLM_KWARGS, recommender=secondary
    )
    llm_rec = recommender.initial_recommender

    assert select_recommender(recommender, 0) is llm_rec
    assert select_recommender(recommender, 1) is secondary

    recommender2 = roundtrip(recommender)
    assert recommender2 == recommender
    rec = select_recommender(recommender2, 1)
    assert rec == secondary
