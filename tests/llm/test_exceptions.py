"""Tests for the LLM exception import surface."""

import pytest

import baybe.exceptions

_INTERNAL_LLM_EXCEPTIONS = [
    "LLMCallError",
    "LLMAuthenticationError",
    "LLMResponseError",
    "LLMMalformedResponseError",
    "LLMUnknownParameterError",
    "LLMBatchConstraintValidationError",
    "LLMIneligiblePointsError",
    "LLMDuplicateRecommendationsError",
    "LLMBatchSizeError",
]


@pytest.mark.parametrize("name", _INTERNAL_LLM_EXCEPTIONS)
def test_internal_llm_exceptions_are_not_public(name: str):
    """The LLM-specific exception subclasses stay out of the public module."""
    assert not hasattr(baybe.exceptions, name)
