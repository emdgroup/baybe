"""JSON extraction from language model responses."""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any


def extract_json_array(response: str, /) -> list[dict[str, Any]]:
    """Extract the JSON array payload from a raw language model response.

    Models often wrap the array in Markdown fences or prose, or emit several blocks
    (e.g. after reconsidering). Return the *last* array that is a list of objects (the
    intended answer), ignoring prose and stray brackets like ``x[0]``.

    Args:
        response: The raw response text.

    Returns:
        The decoded JSON array of objects.

    Raises:
        ValueError: If no JSON array of objects is found in the response.
    """
    decoder = json.JSONDecoder()
    last_object_array: list[dict[str, Any]] | None = None
    search_start = 0
    while (start := response.find("[", search_start)) != -1:
        try:
            value, end = decoder.raw_decode(response, start)
        except JSONDecodeError:
            search_start = start + 1  # not a valid array here; try the next "["
            continue
        is_object_list = (
            isinstance(value, list)
            and bool(value)
            and all(isinstance(x, dict) for x in value)
        )
        if is_object_list:
            last_object_array = value
        search_start = end  # keep scanning; a later block wins
    if last_object_array is None:
        raise ValueError("No JSON array of objects found in the response.")
    return last_object_array
