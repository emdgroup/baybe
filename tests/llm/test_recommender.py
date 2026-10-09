"""Tests for the LLM recommender integration."""

import warnings
from unittest.mock import patch

import pandas as pd
import pytest
from pytest import param

from baybe import Campaign
from baybe._optional.info import LLM_INSTALLED
from baybe._optional.llm import AuthenticationError
from baybe.exceptions import (
    LLMAuthenticationError,
    LLMCallError,
    LLMMalformedResponseError,
    LLMResponseError,
)
from baybe.objectives import SingleTargetObjective
from baybe.recommenders.pure.llm._prompts import make_prompt
from baybe.recommenders.pure.llm.llm import LLMRecommender
from baybe.targets import NumericalTarget
from tests.llm._mock import (
    PATCH_TARGET,
    make_continuous_searchspace,
    make_discrete_searchspace,
    make_hybrid_searchspace,
    make_objective,
    make_valid_json,
    mock_response,
)

if not LLM_INSTALLED:
    pytest.skip("LLM dependencies not installed.", allow_module_level=True)

_RECOMMENDER = LLMRecommender(
    model="test/model", experiment_description="Test experiment."
)
_OBJECTIVE = make_objective()


@pytest.mark.parametrize(
    "searchspace_factory",
    [
        param(make_discrete_searchspace, id="discrete"),
        param(make_continuous_searchspace, id="continuous"),
        param(make_hybrid_searchspace, id="hybrid"),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 3], ids=["single", "batch"])
def test_recommend_happy_path(searchspace_factory, batch_size):
    """The recommend flow returns a DataFrame with correct shape.

    Value correctness is covered by the parsing tests; this test verifies
    the end-to-end flow (prompt building, model call, parsing).
    """
    searchspace = searchspace_factory()
    with patch(PATCH_TARGET) as mock:
        mock.return_value = mock_response(make_valid_json(searchspace, batch_size))
        result = _RECOMMENDER.recommend(batch_size, searchspace, _OBJECTIVE)
    assert len(result) == batch_size
    assert set(result.columns) == {p.name for p in searchspace.parameters}


def test_completion_arguments_forwarded():
    """The model, prompt, and litellm_args are forwarded to the completion call."""
    searchspace = make_discrete_searchspace()
    recommender = LLMRecommender(
        model="test/my-model",
        experiment_description="Test.",
        litellm_args={"temperature": 0.5, "max_tokens": 100},
    )
    with patch(PATCH_TARGET) as mock:
        mock.return_value = mock_response(make_valid_json(searchspace, 1))
        recommender.recommend(1, searchspace, _OBJECTIVE)

    mock.assert_called_once()
    kwargs = mock.call_args.kwargs
    assert kwargs["model"] == "test/my-model"
    expected_prompt = make_prompt(
        1, searchspace, _OBJECTIVE, experiment_description="Test."
    )
    assert kwargs["messages"] == [{"role": "user", "content": expected_prompt}]
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 100


_SEARCHSPACE = make_discrete_searchspace()
_AUTH_ERROR = AuthenticationError(
    message="bad key", llm_provider="test", model="test/model"
)


@pytest.mark.parametrize(
    ("exception", "expected_error"),
    [
        param(
            _AUTH_ERROR,
            LLMAuthenticationError,
            id="auth",
        ),
        param(
            ConnectionError("timeout"),
            LLMCallError,
            id="provider",
        ),
    ],
)
def test_call_errors(exception, expected_error):
    """Provider-level failures are wrapped and preserve the original cause."""
    with patch(PATCH_TARGET, side_effect=exception) as mock:
        with pytest.raises(expected_error):
            _RECOMMENDER.recommend(1, _SEARCHSPACE, _OBJECTIVE)
    assert mock.call_count == 1


def test_empty_response():
    """A response with ``None`` content raises without a recovery attempt."""
    with patch(PATCH_TARGET, return_value=mock_response(None)) as mock:
        with pytest.raises(LLMResponseError):
            _RECOMMENDER.recommend(1, _SEARCHSPACE, _OBJECTIVE)
    assert mock.call_count == 1


@pytest.mark.parametrize(
    ("batch_size", "recovery_response", "expected_error"),
    [
        param(
            1,
            mock_response(make_valid_json(_SEARCHSPACE, 1)),
            None,
            id="success_single",
        ),
        param(
            3,
            mock_response(make_valid_json(_SEARCHSPACE, 3)),
            None,
            id="success_batch",
        ),
        param(
            1,
            mock_response("not valid json"),
            LLMResponseError,
            id="bad_again",
        ),
        param(
            1,
            _AUTH_ERROR,
            LLMAuthenticationError,
            id="auth_on_recovery",
        ),
        param(
            1,
            ConnectionError("timeout"),
            LLMCallError,
            id="provider_on_recovery",
        ),
    ],
)
def test_recovery(batch_size, recovery_response, expected_error):
    """After a bad first response, a recovery attempt is made exactly once."""
    side_effect = [mock_response("not valid json"), recovery_response]
    with patch(PATCH_TARGET, side_effect=side_effect) as mock:
        if expected_error is None:
            _RECOMMENDER.recommend(batch_size, _SEARCHSPACE, _OBJECTIVE)
        else:
            with pytest.raises(expected_error):
                _RECOMMENDER.recommend(batch_size, _SEARCHSPACE, _OBJECTIVE)
    assert mock.call_count == 2


def test_recovery_prompt_content():
    """The recovery prompt carries the full context and the error details."""
    bad_content = "not valid json"
    valid_content = make_valid_json(_SEARCHSPACE, 1)
    side_effect = [mock_response(bad_content), mock_response(valid_content)]
    with patch(PATCH_TARGET, side_effect=side_effect) as mock:
        _RECOMMENDER.recommend(1, _SEARCHSPACE, _OBJECTIVE)

    prompt_kwargs = dict(
        batch_size=1,
        searchspace=_SEARCHSPACE,
        objective=_OBJECTIVE,
        experiment_description=_RECOMMENDER.experiment_description,
    )
    expected_first = make_prompt(**prompt_kwargs)
    expected_recovery = make_prompt(
        **prompt_kwargs,
        error=LLMMalformedResponseError("Response contains no JSON array of objects."),
        original_response=bad_content,
    )

    actual_first = mock.call_args_list[0].kwargs["messages"][0]["content"]
    actual_recovery = mock.call_args_list[1].kwargs["messages"][0]["content"]
    assert actual_first == expected_first
    assert actual_recovery == expected_recovery


@pytest.mark.parametrize(
    ("objective", "warns"),
    [
        param(
            SingleTargetObjective(target=NumericalTarget("yield")),
            True,
            id="no_metadata",
        ),
        param(
            _OBJECTIVE,
            False,
            id="with_metadata",
        ),
    ],
)
def test_objective_metadata_warning(objective, warns):
    """A warning is raised when the objective has no metadata description."""
    searchspace = make_discrete_searchspace()
    with patch(PATCH_TARGET) as mock:
        mock.return_value = mock_response(make_valid_json(searchspace, 1))
        if warns:
            with pytest.warns(UserWarning, match="no metadata description"):
                _RECOMMENDER.recommend(1, searchspace, objective)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                _RECOMMENDER.recommend(1, searchspace, objective)


def test_context_stash_replacement():
    """The stashed context reflects the current call, not a previous one."""
    searchspace = make_discrete_searchspace()
    first_measurements = pd.DataFrame(
        {"Cat": ["A"], "Num": [1.0], "Solvent": ["Water"], "yield": [10.0]}
    )
    second_measurements = pd.DataFrame(
        {"Cat": ["B"], "Num": [2.0], "Solvent": ["Ethanol"], "yield": [20.0]}
    )

    with patch(PATCH_TARGET) as mock:
        mock.return_value = mock_response(make_valid_json(searchspace, 1))
        _RECOMMENDER.recommend(
            1, searchspace, _OBJECTIVE, measurements=first_measurements
        )
        _RECOMMENDER.recommend(
            1, searchspace, _OBJECTIVE, measurements=second_measurements
        )

    first_prompt = mock.call_args_list[0].kwargs["messages"][0]["content"]
    second_prompt = mock.call_args_list[1].kwargs["messages"][0]["content"]

    expected_first = make_prompt(
        1,
        searchspace,
        _OBJECTIVE,
        measurements=first_measurements,
        experiment_description=_RECOMMENDER.experiment_description,
    )
    expected_second = make_prompt(
        1,
        searchspace,
        _OBJECTIVE,
        measurements=second_measurements,
        experiment_description=_RECOMMENDER.experiment_description,
    )
    assert first_prompt == expected_first
    assert second_prompt == expected_second


def test_recommend_with_campaign():
    """Stateful recommendation through a Campaign works across iterations."""
    searchspace = make_discrete_searchspace()
    campaign = Campaign(
        searchspace=searchspace,
        objective=_OBJECTIVE,
        recommender=LLMRecommender(model="test/model", experiment_description="Test."),
        allow_recommending_already_recommended=True,
    )

    with patch(PATCH_TARGET) as mock:
        mock.return_value = mock_response(make_valid_json(searchspace, 1))
        rec1 = campaign.recommend(batch_size=1)

    rec1["yield"] = 42.0
    campaign.add_measurements(rec1)

    with patch(PATCH_TARGET) as mock:
        mock.return_value = mock_response(make_valid_json(searchspace, 1))
        campaign.recommend(batch_size=1)
