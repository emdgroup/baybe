"""Tests for the shared LLM query utility."""

from unittest.mock import patch

import pytest
from pytest import param

from baybe._llm.exceptions import LLMAuthenticationError, LLMCallError
from baybe._llm.query import query_model
from baybe._optional.info import LLM_INSTALLED
from tests.llm._mock import PATCH_TARGET

if not LLM_INSTALLED:
    pytest.skip("LLM dependencies not installed.", allow_module_level=True)


from baybe._optional.llm import AuthenticationError

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
            query_model("Prompt.", model="test/model")
    assert mock.call_count == 1


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
