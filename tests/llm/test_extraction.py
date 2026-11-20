"""Tests for the extraction of structured payloads from language model responses."""

import gc

import pytest
from attrs import define

from baybe._llm.exceptions import LLMMalformedResponseError
from baybe._llm.extraction import extract_payload


@define
class _Item:
    name: str
    count: int


@define
class _Payload:
    items: list[_Item]


_PAYLOAD = '{"items": [{"name": "a", "count": 1}]}'


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(_PAYLOAD, id="bare"),
        pytest.param(f"```json\n{_PAYLOAD}\n```", id="markdown_fences"),
        pytest.param(f"Use {{braces}} as prose. Result: {_PAYLOAD} Done.", id="prose"),
        pytest.param(f'{{"items": []}} Actually: {_PAYLOAD}', id="last_object_wins"),
        pytest.param(f'{_PAYLOAD} {{"broken": ', id="trailing_invalid_json"),
        pytest.param(
            f'{_PAYLOAD} Confidence: {{"confidence": 0.8}}', id="trailing_metadata"
        ),
        pytest.param(
            f'{_PAYLOAD} Note: I kept {{"name": "a"}} fixed.', id="quoted_parameter"
        ),
        pytest.param(
            f'{_PAYLOAD} This respects {{"type": "C", "parameters": ["name"]}}.',
            id="echoed_input",
        ),
        pytest.param(
            '{"items": []} {"items": [{"name": "a", "count": 1}], "confidence": 0.9}',
            id="later_compatible_block_with_extra_key",
        ),
        pytest.param(
            f'```json\n{_PAYLOAD}\n```\n```json\n{{"summary": "x"}}\n```',
            id="trailing_fenced_block",
        ),
        pytest.param(
            f'```json\n{{"summary": "x"}}\n```\n```json\n{_PAYLOAD}\n```',
            id="leading_incompatible_block",
        ),
    ],
)
def test_extract_payload(response):
    """The last schema-compatible JSON object is structured into the schema."""
    assert extract_payload(response, _Payload) == _Payload([_Item("a", 1)])


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(
            '{"items": [{"name": "a", "count": 1}], "note": "x"}', id="envelope"
        ),
        pytest.param('{"items": [{"name": "a", "count": 1, "note": "x"}]}', id="item"),
    ],
)
def test_extract_payload_ignores_extra_keys(response):
    """Extra keys beyond the schema are ignored."""
    assert extract_payload(response, _Payload) == _Payload([_Item("a", 1)])


@pytest.mark.parametrize(
    ("response", "match", "problems"),
    [
        pytest.param("No JSON here.", "contains no JSON object", (), id="no_json"),
        pytest.param(
            '{"items": [{}]}',
            "expected format",
            (
                "required field missing @ $.items[0].name",
                "required field missing @ $.items[0].count",
            ),
            id="missing_keys",
        ),
        pytest.param(
            '{"items": [{"name": "a", "counts": 1}]}',
            "expected format",
            ("required field missing @ $.items[0].count",),
            id="misspelled_key",
        ),
        pytest.param(
            '{"items": "x"} Note: {"items": [{"name": "a"}]}',
            "expected format",
            ("required field missing @ $.items[0].count",),
            id="none_compatible",
        ),
    ],
)
def test_extract_payload_raises_on_invalid(response, match, problems):
    """Invalid responses are rejected, reporting each schema violation for recovery."""
    with pytest.raises(LLMMalformedResponseError, match=match) as exc_info:
        extract_payload(response, _Payload)
    error = exc_info.value
    assert error.problems == problems
    for problem in problems:
        assert problem in error.recovery_instruction


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
