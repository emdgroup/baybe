"""Extraction of structured payloads from language model responses."""

from __future__ import annotations

import json
import re
from typing import TypeVar

import cattrs

from baybe._llm.exceptions import LLMMalformedResponseError

_T = TypeVar("_T")

_CONVERTER = cattrs.Converter(forbid_extra_keys=True)


def extract_payload(response: str, schema: type[_T], /) -> _T:
    """Structure the last JSON object in a response into the given schema.

    Args:
        response: The raw response text.
        schema: The type describing the expected payload.

    Returns:
        The structured payload.

    Raises:
        LLMMalformedResponseError: If the response contains no JSON object or the
            last one does not match the schema.
    """
    decoder = json.JSONDecoder()
    payload, end = None, 0
    for match in re.finditer(r"\{", response):
        if match.start() < end:
            continue
        try:
            payload, end = decoder.raw_decode(response, match.start())
        except json.JSONDecodeError:
            continue

    if payload is None:
        raise LLMMalformedResponseError("The response contains no JSON object.")
    try:
        return _CONVERTER.structure(payload, schema)
    except cattrs.BaseValidationError as ex:
        # NOTE: cattrs raises a nested exception group with an uninformative message.
        #   Flattening it yields located problems that tell the model what to fix.
        problems = cattrs.transform_error(ex)
        raise LLMMalformedResponseError(
            f"The response does not match the expected format: {'; '.join(problems)}",
            problems=problems,
        ) from ex
