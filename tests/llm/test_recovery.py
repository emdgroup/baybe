"""Tests for the shared LLM recovery loop."""

from functools import partial
from unittest.mock import Mock, call

import pytest
from pytest import param

from baybe._llm.exceptions import (
    LLMAuthenticationError,
    LLMCallError,
    LLMMalformedResponseError,
    LLMResponseError,
)
from baybe._llm.recovery import query_with_recovery

_USABLE = "usable"
_UNUSABLE = "unusable"


def _parse(content: str) -> str:
    """Accept the usable marker and reject everything else."""
    if content == _USABLE:
        return "parsed"
    raise LLMResponseError(f"Unusable content: {content!r}.")


def _make_loop(outcomes):
    """Return a runner and its mocks for the recovery loop."""
    query = Mock(side_effect=list(outcomes))
    make_recovery_prompt = Mock(return_value="recovery")
    run = partial(
        query_with_recovery,
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
        param((_UNUSABLE, _USABLE), LLMResponseError, _UNUSABLE, id="parse_error"),
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
        LLMResponseError, match="even after a recovery attempt"
    ) as exc_info:
        run()
    assert query.call_count == 2
    assert make_recovery_prompt.call_args.args[1] == _UNUSABLE
    assert isinstance(exc_info.value.__cause__, LLMResponseError)
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
