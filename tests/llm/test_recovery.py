"""Tests for the shared LLM recovery loop."""

from unittest.mock import Mock

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


@pytest.mark.parametrize(
    ("outcomes", "expected_result", "expected_error", "expected_calls"),
    [
        param(
            (LLMMalformedResponseError("No content."), _USABLE),
            "parsed",
            None,
            2,
            id="malformed_query_then_ok",
        ),
        param(
            (_UNUSABLE, _USABLE),
            "parsed",
            None,
            2,
            id="parse_error_then_ok",
        ),
        param(
            (_UNUSABLE, _UNUSABLE),
            None,
            LLMResponseError,
            2,
            id="error_twice",
        ),
        param(
            (LLMCallError("timeout"),),
            None,
            LLMCallError,
            1,
            id="call_error_first",
        ),
        param(
            (_UNUSABLE, LLMAuthenticationError("bad key")),
            None,
            LLMAuthenticationError,
            2,
            id="auth_on_recovery",
        ),
        param(
            (_UNUSABLE, LLMCallError("timeout")),
            None,
            LLMCallError,
            2,
            id="call_error_on_recovery",
        ),
    ],
)
def test_recovery(outcomes, expected_result, expected_error, expected_calls):
    """The response is retried exactly once; call errors are never retried."""
    query = Mock(side_effect=list(outcomes))
    if expected_error is None:
        result = query_with_recovery(
            "prompt",
            query=query,
            parse=_parse,
            make_recovery_prompt=lambda error, content: "recovery",
        )
        assert result == expected_result
    else:
        with pytest.raises(expected_error):
            query_with_recovery(
                "prompt",
                query=query,
                parse=_parse,
                make_recovery_prompt=lambda error, content: "recovery",
            )
    assert query.call_count == expected_calls
