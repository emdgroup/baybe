"""Tests for the LLM exception import surface."""

import pytest

import baybe._llm.exceptions as llm_exceptions
import baybe.exceptions

_LLM_EXCEPTIONS = [
    "LLMError",
    "LLMCallError",
    "LLMAuthenticationError",
    "LLMResponseError",
    "LLMMalformedResponseError",
    "LLMUnknownParameterError",
    "LLMBatchConstraintValidationError",
    "LLMIneligiblePointsError",
    "LLMDuplicateRecommendationsError",
    "LLMBatchSizeError",
    "LLMResponseWarning",
]

_INTERNAL_LLM_EXCEPTIONS = [name for name in _LLM_EXCEPTIONS if name != "LLMError"]


def test_llm_error_public_reexport():
    """``LLMError`` is re-exported from the public exceptions module."""
    assert baybe.exceptions.LLMError is llm_exceptions.LLMError


@pytest.mark.parametrize("name", _LLM_EXCEPTIONS)
def test_llm_exceptions_defined_internal(name: str):
    """All LLM-specific classes are defined in the internal exceptions module."""
    assert getattr(llm_exceptions, name).__module__ == llm_exceptions.__name__


@pytest.mark.parametrize("name", _INTERNAL_LLM_EXCEPTIONS)
def test_only_llm_error_is_public(name: str):
    """Only ``LLMError`` remains available from the public exceptions module."""
    assert not hasattr(baybe.exceptions, name)
