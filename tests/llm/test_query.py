"""Tests for the shared LLM querying and recovery utilities."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
from pytest import param

from baybe._llm.exceptions import LLMMalformedResponseError
from baybe._llm.query import query_model, query_model_with_recovery
from baybe._optional.info import LLM_INSTALLED
from baybe.exceptions import (
    LLMAuthenticationError,
    LLMCallError,
    LLMRecoveryFailedError,
)
from tests.llm._mock import PATCH_TARGET

if not LLM_INSTALLED:
    pytest.skip("LLM dependencies not installed.", allow_module_level=True)


from baybe._optional.llm import AuthenticationError

_AUTH_ERROR = AuthenticationError(
    message="bad key", llm_provider="test", model="test/model"
)
_USABLE = "usable"
_UNUSABLE = "unusable"


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
        with pytest.raises(expected_error) as exc_info:
            query_model("Prompt.", model="test/model")
    assert mock.call_count == 1
    assert exc_info.value.__cause__ is exception


@pytest.mark.parametrize(
    ("response", "cause_type"),
    [
        param(SimpleNamespace(), AttributeError, id="missing_choices"),
        param(SimpleNamespace(choices=[]), IndexError, id="empty_choices"),
    ],
)
def test_malformed_response_structure(response, cause_type):
    """A response without usable choices is reported as malformed."""
    with patch(PATCH_TARGET, return_value=response):
        with pytest.raises(
            LLMMalformedResponseError, match="unexpected response structure"
        ) as exc_info:
            query_model("Prompt.", model="test/model")
    assert isinstance(exc_info.value.__cause__, cause_type)


@pytest.mark.parametrize("reserved_key", ["model", "messages", "stream"])
def test_reserved_litellm_args_rejected(reserved_key):
    """Reserved LiteLLM arguments are rejected before any provider call."""
    with patch(PATCH_TARGET) as mock:
        with pytest.raises(ValueError, match="controlled by BayBE"):
            query_model(
                "Prompt.",
                model="test/model",
                litellm_args={reserved_key: True},
            )
    mock.assert_not_called()


def _parse(content: str) -> str:
    """Accept the usable marker and reject everything else."""
    if content == _USABLE:
        return "parsed"
    raise LLMMalformedResponseError(f"Unusable content: {content!r}.")


def _make_loop(outcomes):
    """Return a runner and its mocks for the recovery loop."""
    query = Mock(side_effect=list(outcomes))
    make_recovery_prompt = Mock(return_value="recovery")
    run = partial(
        query_model_with_recovery,
        "prompt",
        query=query,
        parse=_parse,
        make_recovery_prompt=make_recovery_prompt,
    )
    return run, query, make_recovery_prompt


def test_usable_response_is_returned():
    """A usable first response is returned without a recovery attempt."""
    run, query, make_recovery_prompt = _make_loop([_USABLE])
    assert run() == "parsed"
    query.assert_called_once_with("prompt")
    make_recovery_prompt.assert_not_called()


@pytest.mark.parametrize(
    ("outcomes", "recovery_error_type", "previous_content"),
    [
        param(
            (_UNUSABLE, _USABLE),
            LLMMalformedResponseError,
            _UNUSABLE,
            id="parse_error",
        ),
        param(
            (LLMMalformedResponseError("No content."), _USABLE),
            LLMMalformedResponseError,
            None,
            id="query_error",
        ),
    ],
)
def test_unusable_response_is_retried(outcomes, recovery_error_type, previous_content):
    """An unusable response triggers exactly one corrective query."""
    run, query, make_recovery_prompt = _make_loop(outcomes)
    assert run() == "parsed"
    assert query.call_args_list == [call("prompt"), call("recovery")]
    error, content = make_recovery_prompt.call_args.args
    assert isinstance(error, recovery_error_type)
    assert content == previous_content


def test_unusable_response_twice_is_wrapped():
    """Two unusable responses raise a wrapped error caused by the second one."""
    run, query, make_recovery_prompt = _make_loop([_UNUSABLE, _UNUSABLE])
    with pytest.raises(
        LLMRecoveryFailedError, match="even after a recovery attempt"
    ) as exc_info:
        run()
    assert query.call_count == 2
    assert make_recovery_prompt.call_args.args[1] == _UNUSABLE
    assert isinstance(exc_info.value.__cause__, LLMMalformedResponseError)
    assert "Initial problem" in str(exc_info.value)
    assert "Remaining problem after recovery" in str(exc_info.value)


def test_call_error_before_recovery_is_not_retried():
    """A call error on the first query is propagated without a retry."""
    run, query, make_recovery_prompt = _make_loop([LLMCallError("timeout")])
    with pytest.raises(LLMCallError):
        run()
    query.assert_called_once_with("prompt")
    make_recovery_prompt.assert_not_called()


@pytest.mark.parametrize(
    ("outcomes", "expected_error"),
    [
        param(
            (_UNUSABLE, LLMAuthenticationError("bad key")),
            LLMAuthenticationError,
            id="authentication",
        ),
        param((_UNUSABLE, LLMCallError("timeout")), LLMCallError, id="call"),
    ],
)
def test_call_error_during_recovery_propagates(outcomes, expected_error):
    """A call error during the corrective query is propagated."""
    run, query, make_recovery_prompt = _make_loop(outcomes)
    with pytest.raises(expected_error):
        run()
    assert query.call_count == 2
    assert make_recovery_prompt.call_args.args[1] == _UNUSABLE
