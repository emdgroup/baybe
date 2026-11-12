"""JSON extraction from language model responses."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from json import JSONDecodeError
from typing import Any, TypeVar

import cattrs

from baybe._llm.exceptions import LLMMalformedResponseError

_T = TypeVar("_T")

_BRACKET_PATTERN = re.compile(r"[\[{]")
"""Pattern matching the start of a JSON array or object."""


def extract_json(
    response: str,
    /,
    accept: Callable[[Any], bool],
) -> Any:
    """Extract the last JSON array or object accepted by ``accept``.

    Decoded values are skipped as a whole, so brackets inside strings are ignored;
    rejected containers are searched through their decoded children.

    Args:
        response: The raw response text.
        accept: Predicate selecting the intended payload.

    Returns:
        The last accepted value.

    Raises:
        ValueError: If no accepted value is found.
    """
    decoder = json.JSONDecoder()
    last_accepted: Any = None
    skip_until = 0
    for match in _BRACKET_PATTERN.finditer(response):
        start = match.start()
        if start < skip_until:
            continue
        try:
            value, end = decoder.raw_decode(response, start)
        except JSONDecodeError:
            continue  # not valid JSON here; try the next candidate

        skip_until = end  # never scan the decoded value's raw text again
        candidates = [value]
        while candidates:
            candidate = candidates.pop()
            if accept(candidate):
                last_accepted = candidate
                continue  # accepted values are atomic
            children = candidate.values() if isinstance(candidate, dict) else candidate
            candidates.extend(
                child
                for child in reversed(list(children))
                if isinstance(child, (dict, list))
            )
    # Candidates are always containers, so ``None`` can only mean "nothing accepted."
    if last_accepted is None:
        raise ValueError("No accepted JSON value found in the response.")
    return last_accepted


def structure_json(
    value: Any,
    cl: type[_T],
    /,
    *,
    forbid_extra_keys: bool = False,
) -> _T:
    """Structure a decoded JSON value into the given type.

    Args:
        value: The decoded JSON value.
        cl: The type to structure the value into.
        forbid_extra_keys: Whether to reject keys not defined by the target type.
            By default, unknown keys are ignored.

    Returns:
        The structured value.

    Raises:
        LLMMalformedResponseError: If the value does not match the expected structure.
    """
    structure = (
        cattrs.Converter(forbid_extra_keys=True).structure
        if forbid_extra_keys
        else cattrs.structure
    )
    try:
        return structure(value, cl)
    except Exception as exc:
        raise LLMMalformedResponseError(
            f"The language model response does not match the expected format. {exc}"
        ) from exc
