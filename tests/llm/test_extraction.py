"""Tests for JSON extraction from language model responses."""

import gc

import pytest
from attrs import define

from baybe._llm.exceptions import LLMMalformedResponseError
from baybe._llm.extraction import extract_json, structure_json
from baybe.recommenders.pure.llm._parsing import _is_object_array


@pytest.mark.parametrize(
    ("response", "accept", "expected"),
    [
        pytest.param(
            '[{"a": 1}, {"b": 2}]',
            _is_object_array,
            [{"a": 1}, {"b": 2}],
            id="bare",
        ),
        pytest.param(
            '```json\n[{"a": 1}]\n```',
            _is_object_array,
            [{"a": 1}],
            id="markdown_fences",
        ),
        pytest.param(
            'Here are my recommendations: [{"a": 1}]',
            _is_object_array,
            [{"a": 1}],
            id="prose_before",
        ),
        pytest.param(
            '[{"first": 1}]\nActually, let me reconsider:\n[{"second": 2}]',
            _is_object_array,
            [{"second": 2}],
            id="multi_block_last_wins",
        ),
        pytest.param(
            '[{"a": 1}]\nSome text\n[1, 2, 3]',
            _is_object_array,
            [{"a": 1}],
            id="object_array_before_scalar_array",
        ),
        pytest.param(
            '[{"a": [1, 2]}, {"b": 3}]',
            _is_object_array,
            [{"a": [1, 2]}, {"b": 3}],
            id="nested",
        ),
        pytest.param(
            '{"recommendations": [{"a": 1}]}',
            _is_object_array,
            [{"a": 1}],
            id="object_wrapping_array",
        ),
        pytest.param(
            '[[{"a": 1}]]',
            _is_object_array,
            [{"a": 1}],
            id="nested_arrays",
        ),
        pytest.param(
            'Use {braces} as prose. Result: [{"a": 1}]',
            _is_object_array,
            [{"a": 1}],
            id="braces_in_prose",
        ),
        pytest.param(
            '{"result": [1, 2], "note": "[3, 4]"}',
            lambda value: isinstance(value, list),
            [1, 2],
            id="string_literal_array_ignored",
        ),
        pytest.param(
            '[{"a": 1}]\n["note about [{}]"]',
            _is_object_array,
            [{"a": 1}],
            id="array_in_later_string_ignored",
        ),
        pytest.param(
            '{"recommendations": [{"x": 1}], "note": "e.g. [{}]"}',
            _is_object_array,
            [{"x": 1}],
            id="wrapper_payload_beats_string_example",
        ),
        pytest.param(
            '{"outer": {"inner": 1}}',
            lambda value: isinstance(value, dict),
            {"outer": {"inner": 1}},
            id="nested_object_outer_wins",
        ),
        pytest.param(
            '[[{"a": 1}]]',
            lambda value: isinstance(value, list),
            [[{"a": 1}]],
            id="outermost_array_wins",
        ),
        pytest.param(
            '{"other": 1}\n{"answer": 42}',
            lambda value: isinstance(value, dict) and "answer" in value,
            {"answer": 42},
            id="dict_with_required_key",
        ),
        pytest.param(
            "{}",
            lambda value: isinstance(value, dict),
            {},
            id="empty_object",
        ),
    ],
)
def test_extract_json(response, accept, expected):
    """The extractor returns the last accepted JSON value."""
    assert extract_json(response, accept) == expected


@pytest.mark.parametrize(
    ("response", "accept"),
    [
        pytest.param("[]", _is_object_array, id="empty_array"),
        pytest.param("[1, 2, 3]", _is_object_array, id="array_of_scalars"),
        pytest.param("No JSON here at all.", _is_object_array, id="no_array"),
        pytest.param('[{"a": 1}, 2]', _is_object_array, id="mixed_objects_and_scalars"),
        pytest.param(
            '[{"a": 1}, {"b": 2}', _is_object_array, id="incomplete_array_opening"
        ),
        pytest.param(
            '{"a": 1}, {"b": 2}]', _is_object_array, id="incomplete_array_closing"
        ),
        pytest.param(
            '{"a": 1}', lambda value: isinstance(value, list), id="dict_not_list"
        ),
        pytest.param(
            '["see {", ": 1}"]',
            lambda value: isinstance(value, dict),
            id="no_dict_stitched_from_strings",
        ),
        pytest.param(
            "The answer is 42.",
            lambda value: isinstance(value, int),
            id="scalars_not_extractable",
        ),
    ],
)
def test_extract_json_raises_on_invalid(response, accept):
    """The extractor raises ``ValueError`` when no accepted value is found."""
    with pytest.raises(ValueError, match="No accepted JSON value"):
        extract_json(response, accept)


@define
class _Item:
    value: int


def test_structure_json():
    """The helper structures a decoded value into the given type."""
    assert structure_json({"value": 1}, _Item) == _Item(value=1)


def test_structure_json_raises_on_invalid():
    """The helper maps structuring failures to a chained format error."""
    with pytest.raises(
        LLMMalformedResponseError, match="does not match the expected format"
    ) as exc_info:
        structure_json({}, _Item)
    assert exc_info.value.__cause__ is not None


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
