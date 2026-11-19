"""Extraction of structured payloads from language model responses."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

import cattrs

from baybe._llm.exceptions import LLMMalformedResponseError

_T = TypeVar("_T")

_CONVERTER = cattrs.Converter(forbid_extra_keys=True)


def extract_payload(response: str, schema: type[_T], /) -> _T:
    """Structure the last schema-compatible JSON object in a response.

    Args:
        response: The raw response text.
        schema: The type describing the expected payload.

    Returns:
        The structured payload.

    Raises:
        LLMMalformedResponseError: If the response contains no JSON object or none
            of them matches the schema.
    """
    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    end = 0
    for match in re.finditer(r"\{", response):
        if match.start() < end:
            continue
        try:
            candidate, end = decoder.raw_decode(response, match.start())
        except json.JSONDecodeError:
            continue
        candidates.append(candidate)

    if not candidates:
        raise LLMMalformedResponseError("The response contains no JSON object.")

    # NOTE: Surrounding text may contain further JSON objects (e.g. metadata or
    #   echoed inputs), so the last object matching the schema is taken. If none
    #   matches, the last object is the most likely intended answer and its problems
    #   are reported.
    errors: list[cattrs.BaseValidationError] = []
    for candidate in reversed(candidates):
        try:
            return _CONVERTER.structure(candidate, schema)
        except cattrs.BaseValidationError as ex:
            errors.append(ex)

    # NOTE: cattrs raises a nested exception group with an uninformative message.
    #   Flattening it yields located problems that tell the model what to fix.
    problems = cattrs.transform_error(errors[0])
    raise LLMMalformedResponseError(
        f"The response does not match the expected format: {'; '.join(problems)}",
        problems=problems,
    ) from errors[0]
