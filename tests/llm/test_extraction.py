"""Tests for JSON extraction from language model responses."""

import json

import pytest

from baybe._llm.extraction import extract_json_array


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        pytest.param(
            '[{"a": 1}, {"b": 2}]',
            '[{"a": 1}, {"b": 2}]',
            id="bare",
        ),
        pytest.param(
            '```json\n[{"a": 1}]\n```',
            '[{"a": 1}]',
            id="markdown_fences",
        ),
        pytest.param(
            'Here are my recommendations: [{"a": 1}]',
            '[{"a": 1}]',
            id="prose_before",
        ),
        pytest.param(
            '[{"first": 1}]\nActually, let me reconsider:\n[{"second": 2}]',
            '[{"second": 2}]',
            id="multi_block_last_wins",
        ),
        pytest.param(
            '[{"a": 1}]\nSome text\n[1, 2, 3]',
            '[{"a": 1}]',
            id="object_array_before_scalar_array",
        ),
        pytest.param(
            '[{"a": [1, 2]}, {"b": 3}]',
            '[{"a": [1, 2]}, {"b": 3}]',
            id="nested",
        ),
    ],
)
def test_extract_json_array(response, expected):
    """The extractor finds the last JSON array of objects."""
    result = extract_json_array(response)
    assert result == json.loads(expected)


@pytest.mark.parametrize(
    "response",
    [
        pytest.param("[]", id="empty_array"),
        pytest.param("[1, 2, 3]", id="array_of_scalars"),
        pytest.param("No JSON here at all.", id="no_array"),
        pytest.param('[{"a": 1}, 2]', id="mixed_objects_and_scalars"),
        pytest.param('[{"a": 1}, {"b": 2}', id="incomplete_array_opening"),
        pytest.param('{"a": 1}, {"b": 2}]', id="incomplete_array_closing"),
    ],
)
def test_extract_json_array_raises_on_invalid(response):
    """The extractor raises ``ValueError`` when no array of objects is found."""
    with pytest.raises(ValueError, match="No JSON array"):
        extract_json_array(response)
