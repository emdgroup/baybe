"""Validation tests for LLM recommenders."""

import pytest
from pytest import param

from baybe.recommenders import RandomRecommender
from baybe.recommenders.meta.llm import (
    LLMAlternatingRecommender,
    LLMTwoPhaseRecommender,
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
            "'litellm_args' must not contain keys that are set explicitly",
            id="reserved_key_model",
        ),
        param(
            {"litellm_args": {"messages": []}},
            ValueError,
            "'litellm_args' must not contain keys that are set explicitly",
            id="reserved_key_messages",
        ),
        param(
            {"litellm_args": {"api_key": "x"}},
            ValueError,
            "'litellm_args' must not contain credential keys",
            id="credential_key_api_key",
        ),
        param(
            {"litellm_args": {"api_base": "x"}},
            ValueError,
            "'litellm_args' must not contain credential keys",
            id="credential_key_api_base",
        ),
    ],
)
def test_llm_recommender_invalid_construction(overrides, error_cls, match):
    """Invalid arguments to ``LLMRecommender`` raise at construction time."""
    kwargs = {**_VALID_KWARGS, **overrides}
    with pytest.raises(error_cls, match=match):
        LLMRecommender(**kwargs)


@pytest.mark.parametrize(
    ("cls", "kwargs", "error_cls", "match"),
    [
        param(
            LLMTwoPhaseRecommender,
            {
                "initial_recommender": RandomRecommender(),
                "recommender": RandomRecommender(),
            },
            TypeError,
            "instance_of",
            id="twophase_initial_not_llm",
        ),
        param(
            LLMAlternatingRecommender,
            {
                "recommenders": (
                    LLMRecommender(**_VALID_KWARGS),
                    RandomRecommender(),
                    RandomRecommender(),
                ),
                "mode": "cyclic",
            },
            ValueError,
            "exactly two",
            id="alternating_wrong_length",
        ),
        param(
            LLMAlternatingRecommender,
            {
                "recommenders": (RandomRecommender(), RandomRecommender()),
                "mode": "cyclic",
            },
            ValueError,
            "first recommender",
            id="alternating_first_not_llm",
        ),
        param(
            LLMAlternatingRecommender,
            {
                "recommenders": (
                    LLMRecommender(**_VALID_KWARGS),
                    RandomRecommender(),
                ),
                "mode": "raise",
            },
            ValueError,
            "mode",
            id="alternating_wrong_mode",
        ),
    ],
)
def test_llm_meta_recommender_invalid_construction(cls, kwargs, error_cls, match):
    """Invalid arguments to LLM meta recommenders raise at construction time."""
    with pytest.raises(error_cls, match=match):
        cls(**kwargs)


@pytest.mark.parametrize(
    ("cls", "from_model_kwargs", "expected_switch_after"),
    [
        param(
            LLMTwoPhaseRecommender,
            {},
            1,
            id="twophase_defaults",
        ),
        param(
            LLMTwoPhaseRecommender,
            {"switch_after": 5},
            5,
            id="twophase_custom_switch",
        ),
        param(
            LLMAlternatingRecommender,
            {},
            None,
            id="alternating_defaults",
        ),
    ],
)
def test_from_model_constructors(cls, from_model_kwargs, expected_switch_after):
    """The ``from_model`` classmethods wire up the internal structure correctly."""
    secondary = RandomRecommender()
    rec = cls.from_model(
        **_VALID_KWARGS,
        litellm_args={"temperature": 0.5},
        recommender=secondary,
        **from_model_kwargs,
    )

    if cls is LLMTwoPhaseRecommender:
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
