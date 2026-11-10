"""Recovery handling for unusable language model responses."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from baybe._llm.exceptions import LLMResponseError

_T = TypeVar("_T")


def query_with_recovery(
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
        LLMResponseError: If both the initial and the recovery response are unusable.
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
            raise LLMResponseError(
                f"The language model failed to produce a usable response, even "
                f"after a recovery attempt. Initial problem: {initial_error} "
                f"Remaining problem after recovery: {recovery_error}"
            ) from recovery_error
