"""JSON extraction from language model responses."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from json import JSONDecodeError
from typing import Any

_BRACKET_PATTERN = re.compile(r"[\[{]")
"""Pattern matching the start of a JSON array or object."""


def extract_json(
    response: str,
    /,
    accept: Callable[[Any], bool],
) -> Any:
    """Extract a JSON payload from a raw language model response.

    Models often wrap the payload in Markdown fences or prose, or emit several blocks
    (e.g. after reconsidering). Scan for JSON arrays and objects and return the *last*
    value accepted by ``accept``. Rejected values are searched for nested accepted
    values, which handles wrappers such as ``{"recommendations": [...]}``. Accepted
    values are skipped as a whole.

    Args:
        response: The raw response text.
        accept: Predicate deciding whether a decoded JSON value is the intended
            payload.

    Returns:
        The last decoded JSON value accepted by ``accept``.

    Raises:
        ValueError: If no accepted JSON value is found in the response.
    """
    decoder = json.JSONDecoder()
    last_accepted: Any = None
    found = False
    skip_until = 0
    for match in _BRACKET_PATTERN.finditer(response):
        start = match.start()
        if start < skip_until:
            continue
        try:
            value, end = decoder.raw_decode(response, start)
        except JSONDecodeError:
            continue  # not valid JSON here; try the next candidate
        if accept(value):
            last_accepted = value
            found = True
            skip_until = end  # accepted values are skipped as a whole
    if not found:
        raise ValueError("No accepted JSON value found in the response.")
    return last_accepted
