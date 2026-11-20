"""Querying a language model through LiteLLM and recovering from unusable responses."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from baybe._llm.exceptions import LLMMalformedResponseError, LLMResponseError
from baybe.exceptions import (
    LLMAuthenticationError,
    LLMCallError,
    LLMRecoveryFailedError,
)

if TYPE_CHECKING:
    from attrs import Attribute

_T = TypeVar("_T")

RESERVED_LITELLM_KEYS = frozenset({"model", "messages", "stream"})
"""LiteLLM arguments controlled by BayBE that must not be overridden by users.

The model and messages are set explicitly, and the response must not be streamed since
it is parsed as a whole.
"""


def _check_reserved_keys(args: Mapping[str, Any], name: str, /) -> None:
    """Reject LiteLLM arguments that BayBE controls itself.

    Args:
        args: The LiteLLM arguments to check.
        name: The name of the argument mapping, used in the error message.

    Raises:
        ValueError: If at least one reserved key is present.
    """
    if conflicts := RESERVED_LITELLM_KEYS & set(args):
        raise ValueError(
            f"'{name}' must not contain keys that are controlled by "
            f"BayBE itself: {sorted(conflicts)}."
        )


def validate_litellm_args(  # noqa: DOC101, DOC103
    _: Any, attribute: Attribute, value: Any
) -> None:
    """Validate that LiteLLM arguments do not contain reserved keys."""
    _check_reserved_keys(value, attribute.alias or attribute.name)


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
        ValueError: If ``litellm_args`` contains a reserved key.
        LLMAuthenticationError: If authentication with the provider fails.
        LLMCallError: If the call fails for another reason (e.g. network, rate
            limiting, timeout, unknown model) before a response is produced.
        LLMMalformedResponseError: If a response is returned but has an unexpected
            structure or no content.
    """
    _check_reserved_keys(litellm_args or {}, "litellm_args")

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
    #   `query_model` rejects streaming via `RESERVED_LITELLM_KEYS` above, and the
    #   `except` below guards against any other unexpected response type at runtime.
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


def query_model_with_recovery(
    prompt: str,
    /,
    *,
    query: Callable[[str], str],
    parse: Callable[[str], _T],
    make_recovery_prompt: Callable[[LLMResponseError, str | None], str],
) -> _T:
    """Query a language model and recover once from an unusable response.

    The query is repeated exactly once if the first response cannot be used. Call
    errors are never retried.

    Args:
        prompt: The initial prompt to send to the language model.
        query: Callable that sends a prompt to the language model and returns the
            raw response text.
        parse: Callable that turns a raw response into a parsed result and signals an
            unusable response by raising :class:`LLMResponseError`.
        make_recovery_prompt: Callable that builds the corrective prompt from the
            error and the previous raw response. The response is ``None`` if the
            query itself raised.

    Returns:
        The parsed result of the first usable response.

    Raises:
        LLMRecoveryFailedError: If both the initial and the recovery response are
            unusable.
    """
    content: str | None = None
    try:
        content = query(prompt)
        return parse(content)
    except LLMResponseError as initial_error:
        recovery_prompt = make_recovery_prompt(initial_error, content)
        try:
            recovery_content = query(recovery_prompt)
            return parse(recovery_content)
        except LLMResponseError as recovery_error:
            raise LLMRecoveryFailedError(
                f"The language model failed to produce a usable response, even "
                f"after a recovery attempt. Initial problem: {initial_error} "
                f"Remaining problem after recovery: {recovery_error}"
            ) from recovery_error
