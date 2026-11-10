"""Querying a language model through LiteLLM."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from baybe._llm.exceptions import (
    LLMAuthenticationError,
    LLMCallError,
    LLMMalformedResponseError,
)

if TYPE_CHECKING:
    from attrs import Attribute

RESERVED_LITELLM_KEYS = frozenset({"model", "messages", "stream"})
"""LiteLLM arguments controlled by BayBE that must not be overridden by users.

The model and messages are set explicitly, and the response must not be streamed since
it is parsed as a whole.
"""


def validate_litellm_args(  # noqa: DOC101, DOC103
    _: Any, attribute: Attribute, value: Any
) -> None:
    """Validate that LiteLLM arguments do not contain reserved keys."""
    if conflicts := RESERVED_LITELLM_KEYS & set(value):
        raise ValueError(
            f"'{attribute.name}' must not contain keys that are controlled by "
            f"BayBE itself: {sorted(conflicts)}."
        )


def query_model(
    prompt: str,
    /,
    model: str,
    litellm_args: Mapping[str, Any] | None = None,
) -> str:
    """Query a language model and return the raw response text.

    Args:
        prompt: The prompt to send to the language model.
        model: The LiteLLM model identifier to use.
        litellm_args: Additional arguments to pass to LiteLLM (e.g. ``temperature``).

    Returns:
        The raw text content of the model response.

    Raises:
        LLMAuthenticationError: If authentication with the provider fails.
        LLMCallError: If the call fails for another reason (e.g. network, rate
            limiting, timeout, unknown model) before a response is produced.
        LLMMalformedResponseError: If a response is returned but has an unexpected
            structure or no content.
    """
    from baybe._optional.llm import AuthenticationError, completion

    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **(litellm_args or {}),
        )
    except AuthenticationError as e:
        raise LLMAuthenticationError(
            f"Authentication with the language model provider failed "
            f"({type(e).__name__}): {e}. Check the API credentials for model "
            f"'{model}' (e.g. the provider's API key environment variable)."
        ) from e
    except Exception as e:
        raise LLMCallError(
            f"The call to the language model failed ({type(e).__name__}): {e}. "
            f"Check your network connection and the model identifier '{model}'."
        ) from e

    # NOTE: `completion()` is typed to also return a stream, which lacks `.choices`.
    #   Streaming is excluded via `RESERVED_LITELLM_KEYS`, and the `except` below
    #   guards against any other unexpected response type at runtime.
    try:
        choices = response.choices  # pyrefly: ignore[missing-attribute]
        content = choices[0].message.content
    except (AttributeError, IndexError, TypeError) as e:
        raise LLMMalformedResponseError(
            f"The language model returned an unexpected response structure: {e}."
        ) from e

    if content is None:
        raise LLMMalformedResponseError(
            "The language model returned empty content (None)."
        )

    return content
