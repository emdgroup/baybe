"""Validation tests for LLM recommenders."""

import pytest
from pytest import param

from baybe.recommenders import (
    RandomRecommender,
    SequentialMetaRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.recommenders.pure.llm.llm import LLMRecommender

# "Valid" at construction as we do not check those before making any call
_VALID_KWARGS = {"model": "test/model", "experiment_description": "Test experiment."}


@pytest.mark.parametrize(
    ("overrides", "error_cls", "match"),
    [
        param(
            {"model": ""},
            ValueError,
            "Length of 'model' must be >= 1",
            id="empty_model",
        ),
        param(
            {"experiment_description": ""},
            ValueError,
            "Length of 'experiment_description' must be >= 1",
            id="empty_description",
        ),
        param(
            {"litellm_args": {"model": "x"}},
            ValueError,
            "'litellm_args' must not contain keys that are controlled by",
            id="reserved_key_model",
        ),
        param(
            {"litellm_args": {"messages": []}},
            ValueError,
            "'litellm_args' must not contain keys that are controlled by",
            id="reserved_key_messages",
        ),
        param(
            {"litellm_args": {"stream": True}},
            ValueError,
            "'litellm_args' must not contain keys that are controlled by",
            id="reserved_key_stream",
        ),
    ],
)
def test_llm_recommender_invalid_construction(overrides, error_cls, match):
    """Invalid arguments to ``LLMRecommender`` raise at construction time."""
    kwargs = {**_VALID_KWARGS, **overrides}
    with pytest.raises(error_cls, match=match):
        LLMRecommender(**kwargs)


@pytest.mark.parametrize(
    ("constructor", "constructor_kwargs", "expected_switch_after"),
    [
        param(
            TwoPhaseMetaRecommender.with_llm_initial_recommender,
            {},
            1,
            id="twophase_defaults",
        ),
        param(
            TwoPhaseMetaRecommender.with_llm_initial_recommender,
            {"switch_after": 5},
            5,
            id="twophase_custom_switch",
        ),
        param(
            SequentialMetaRecommender.with_alternating_llm_recommender,
            {},
            None,
            id="alternating_defaults",
        ),
    ],
)
def test_llm_meta_recommender_constructors(
    constructor, constructor_kwargs, expected_switch_after
):
    """The LLM meta recommender constructors wire up the internal structure."""
    secondary = RandomRecommender()
    rec = constructor(
        **_VALID_KWARGS,
        litellm_args={"temperature": 0.5},
        recommender=secondary,
        **constructor_kwargs,
    )

    if isinstance(rec, TwoPhaseMetaRecommender):
        llm_rec = rec.initial_recommender
        assert rec.recommender is secondary
        assert rec.switch_after == expected_switch_after
    else:
        llm_rec = rec.recommenders[0]
        assert rec.recommenders[1] is secondary
        assert rec.mode == "cyclic"

    assert llm_rec.model == "test/model"
    assert llm_rec.experiment_description == "Test experiment."
    assert llm_rec.litellm_args == {"temperature": 0.5}
