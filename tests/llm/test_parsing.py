"""Tests for LLM response parsing and validation."""

import json

import numpy as np
import pytest
from attrs import evolve

from baybe.constraints.conditions import ThresholdCondition
from baybe.constraints.continuous import ContinuousLinearConstraint
from baybe.constraints.discrete import (
    DiscreteBatchConstraint,
    DiscreteSelectionConstraint,
)
from baybe.exceptions import (
    LLMConstraintViolationError,
    LLMIneligiblePointsError,
    LLMInvalidParameterValueError,
    LLMMalformedResponseError,
    LLMMissingParameterError,
    LLMNonNumericParameterError,
    LLMResponseWarning,
    LLMUnknownParameterError,
)
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.recommenders.pure.llm._parsing import extract_json_array, parse_llm_response
from baybe.recommenders.pure.llm._schema import (
    _EXPLANATION_FIELD,
    _PARAMETERS_FIELD,
    _response_format,
)
from baybe.searchspace import SearchSpace
from baybe.searchspace._filtered import FilteredSubspaceDiscrete
from tests.llm._mock import (
    make_discrete_searchspace,
    make_response_json,
    make_suggestion,
)


def _make_filtered_searchspace(params, *, forbidden_mask):
    """Create a search space with certain candidates masked as ineligible."""
    ss = SearchSpace.from_product(params)
    filtered = FilteredSubspaceDiscrete.from_subspace(
        ss.discrete, ~np.array(forbidden_mask)
    )
    return evolve(ss, discrete=filtered)


@pytest.fixture(name="discrete_searchspace", scope="module")
def fixture_discrete_searchspace():
    """A discrete search space for tests that do not depend on its contents."""
    return make_discrete_searchspace()


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
            'Here are my suggestions: [{"a": 1}]',
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
    assert json.loads(result) == json.loads(expected)


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
def test_extract_json_array_returns_none(response):
    """The extractor returns None when no array of objects is found."""
    assert extract_json_array(response) is None


@pytest.mark.parametrize("batch_size", [1, 3], ids=["single", "batch"])
def test_response_format(batch_size):
    """The response format example contains exactly ``batch_size`` entries."""
    response_format = _response_format(batch_size)
    assert response_format.count(f'"{_EXPLANATION_FIELD}":') == batch_size
    assert response_format.count(f'"{_PARAMETERS_FIELD}":') == batch_size


@pytest.mark.parametrize(
    ("response", "match"),
    [
        pytest.param(
            "Just some text without JSON.",
            "Response contains no JSON array",
            id="no_json",
        ),
        pytest.param(
            '[{"explanation": "x"}]',
            f"Each suggestion must contain a '{_PARAMETERS_FIELD}' field",
            id="no_parameters",
        ),
        pytest.param(
            '[{"parameters": {"any": 1}}]',
            f"Each suggestion must contain an '{_EXPLANATION_FIELD}' field",
            id="no_explanation",
        ),
        pytest.param(
            '[{"explanation": "x", "parameters": [1, 2]}]',
            "Parameters must be a JSON object",
            id="params_not_dict",
        ),
    ],
)
def test_parse_llm_response_format_errors(discrete_searchspace, response, match):
    """Malformed responses raise ``LLMMalformedResponseError``."""
    with pytest.raises(LLMMalformedResponseError, match=match):
        parse_llm_response(response, discrete_searchspace)


@pytest.mark.parametrize(
    ("searchspace", "overrides", "error_cls", "match"),
    [
        pytest.param(
            SearchSpace.from_product([CategoricalParameter("Cat", ("A", "B"))]),
            {"Unknown": "x"},
            LLMUnknownParameterError,
            "Response contains unknown parameter names",
            id="unknown_param",
        ),
        pytest.param(
            SearchSpace.from_product(
                [NumericalDiscreteParameter("Num", (1.0, 2.0, 3.0))]
            ),
            {"Num": "abc"},
            LLMNonNumericParameterError,
            "non-numeric entries in the provided dataframe",
            id="non_numeric",
        ),
        pytest.param(
            SearchSpace.from_product(
                [NumericalDiscreteParameter("Num", (1.0, 2.0, 3.0))]
            ),
            {"Num": 999},
            LLMInvalidParameterValueError,
            "invalid values in parameter",
            id="out_of_bounds",
        ),
        pytest.param(
            SearchSpace.from_product([CategoricalParameter("Cat", ("A", "B"))]),
            {"Cat": "Z"},
            LLMInvalidParameterValueError,
            "invalid values in parameter",
            id="invalid_category",
        ),
        pytest.param(
            SearchSpace.from_product(
                [NumericalContinuousParameter("x", bounds=(0.0, 10.0))]
            ),
            {"x": 99.0},
            LLMInvalidParameterValueError,
            "invalid values in parameter",
            id="continuous_out_of_bounds",
        ),
    ],
)
def test_parse_llm_response_parameter_errors(searchspace, overrides, error_cls, match):
    """Invalid parameter values raise the corresponding error."""
    response = make_response_json(searchspace, **overrides)
    with pytest.raises(error_cls, match=match):
        parse_llm_response(response, searchspace)


def test_parse_llm_response_missing_parameter():
    """Omitting a required parameter raises ``LLMMissingParameterError``."""
    searchspace = SearchSpace.from_product(
        [
            CategoricalParameter("Cat", ("A", "B")),
            NumericalDiscreteParameter("Num", (1.0, 2.0)),
        ]
    )
    suggestion = make_suggestion(searchspace)
    del suggestion["parameters"]["Cat"]
    response = json.dumps([suggestion])
    with pytest.raises(
        LLMMissingParameterError,
        match="Response is missing values for the following parameters: {'Cat'}",
    ):
        parse_llm_response(response, searchspace)


def test_parse_llm_response_selection_constraint_violation():
    """Violating a discrete selection constraint raises the correct error."""
    constraint = DiscreteSelectionConstraint(
        parameters=["Num"],
        conditions=[ThresholdCondition(threshold=1.5, operator=">=")],
    )
    searchspace = SearchSpace.from_product(
        [
            CategoricalParameter("Cat", ("A", "B")),
            NumericalDiscreteParameter("Num", (1.0, 2.0, 3.0)),
        ],
        constraints=[constraint],
    )
    response = make_response_json(searchspace, Num=1.0)
    with pytest.raises(
        LLMConstraintViolationError, match="DiscreteSelectionConstraint"
    ):
        parse_llm_response(response, searchspace)


def test_parse_llm_response_batch_constraint_violation():
    """Violating a batch constraint raises the correct error."""
    constraint = DiscreteBatchConstraint(parameters=["Cat"])
    searchspace = SearchSpace.from_product(
        [
            CategoricalParameter("Cat", ("A", "B")),
            NumericalDiscreteParameter("Num", (1.0, 2.0)),
        ],
        constraints=[constraint],
    )
    suggestions = [
        {"explanation": "first", "parameters": {"Cat": "A", "Num": 1.0}},
        {"explanation": "second", "parameters": {"Cat": "B", "Num": 2.0}},
    ]
    response = json.dumps(suggestions)
    with pytest.raises(LLMConstraintViolationError, match="DiscreteBatchConstraint"):
        parse_llm_response(response, searchspace)


def test_parse_llm_response_continuous_constraint_warning():
    """Continuous constraints trigger a warning since they cannot be validated."""
    searchspace = SearchSpace.from_product(
        [
            NumericalContinuousParameter("x", bounds=(0.0, 10.0)),
            NumericalContinuousParameter("y", bounds=(0.0, 10.0)),
        ],
        constraints=[
            ContinuousLinearConstraint(
                parameters=["x", "y"],
                coefficients=[1.0, 1.0],
                rhs=10.0,
                operator="<=",
            )
        ],
    )
    response = json.dumps([{"explanation": "test", "parameters": {"x": 3.0, "y": 4.0}}])
    with pytest.warns(LLMResponseWarning, match="cannot be validated"):
        parse_llm_response(response, searchspace)


def test_parse_llm_response_happy_path():
    """A valid response is parsed into a DataFrame with correct shape and values."""
    searchspace = SearchSpace.from_product(
        [
            CategoricalParameter("Cat", ("A", "B", "C")),
            NumericalDiscreteParameter("Num", (1.0, 2.0, 3.0)),
        ]
    )
    suggestions = [
        {"explanation": "first", "parameters": {"Cat": "A", "Num": 1.0}},
        {"explanation": "second", "parameters": {"Cat": "B", "Num": 2.0}},
    ]
    result = parse_llm_response(json.dumps(suggestions), searchspace)
    assert set(result.columns) == {"Cat", "Num"}
    assert list(result["Cat"]) == ["A", "B"]
    assert list(result["Num"]) == [1.0, 2.0]


@pytest.mark.parametrize(
    ("searchspace", "overrides"),
    [
        pytest.param(
            _make_filtered_searchspace(
                [CategoricalParameter("Cat", ("A", "B", "C"))],
                forbidden_mask=[False, True, False],  # B is forbidden
            ),
            {"Cat": "B"},
            id="categorical_forbidden",
        ),
        pytest.param(
            _make_filtered_searchspace(
                [NumericalDiscreteParameter("Num", (1.0, 2.0, 3.0))],
                forbidden_mask=[False, False, True],  # 3.0 is forbidden
            ),
            {"Num": 3.0},
            id="numerical_forbidden",
        ),
        pytest.param(
            _make_filtered_searchspace(
                [NumericalDiscreteParameter("x", (1.0, 2.0, 3.0), tolerance=0.3)],
                forbidden_mask=[True, False, False],  # 1.0 is forbidden
            ),
            {"x": 1.2},
            id="tolerance_snap_forbidden",
        ),
    ],
)
def test_parse_llm_response_ineligible_points(searchspace, overrides):
    """Proposing a forbidden candidate raises ``LLMIneligiblePointsError``."""
    response = make_response_json(searchspace, **overrides)
    with pytest.raises(LLMIneligiblePointsError):
        parse_llm_response(response, searchspace)


def test_parse_llm_response_tolerance_snap():
    """A value within tolerance snaps to the nearest allowed value."""
    searchspace = SearchSpace.from_product(
        [NumericalDiscreteParameter("x", (1.0, 2.0, 3.0), tolerance=0.3)]
    )
    response = json.dumps([{"explanation": "test", "parameters": {"x": 1.2}}])
    result = parse_llm_response(response, searchspace)
    assert result["x"].iloc[0] == 1.0


def test_parse_llm_response_hybrid_output():
    """Hybrid spaces produce output with aligned discrete and continuous parts."""
    searchspace = SearchSpace.from_product(
        [
            CategoricalParameter("Cat", ("A", "B", "C")),
            NumericalContinuousParameter("Cont", bounds=(0.0, 1.0)),
        ]
    )
    suggestions = [
        {"explanation": "first", "parameters": {"Cat": "A", "Cont": 0.3}},
        {"explanation": "second", "parameters": {"Cat": "A", "Cont": 0.7}},
    ]
    result = parse_llm_response(json.dumps(suggestions), searchspace)
    assert set(result.columns) == {"Cat", "Cont"}
    assert list(result["Cat"]) == ["A", "A"]
    assert list(result["Cont"]) == [0.3, 0.7]


def test_parse_llm_response_valid_then_invalid():
    """An error in a later suggestion still raises even if earlier ones are valid."""
    searchspace = SearchSpace.from_product([CategoricalParameter("Cat", ("A", "B"))])
    suggestions = [
        {"explanation": "valid", "parameters": {"Cat": "A"}},
        {"explanation": "invalid", "parameters": {"Cat": "Z"}},
    ]
    with pytest.raises(LLMInvalidParameterValueError):
        parse_llm_response(json.dumps(suggestions), searchspace)


def test_parse_llm_response_partial_missing_parameter():
    """A parameter present in one suggestion but absent in another is caught."""
    searchspace = SearchSpace.from_product(
        [
            CategoricalParameter("Cat", ("A", "B")),
            NumericalDiscreteParameter("Num", (1.0, 2.0)),
        ]
    )
    suggestions = [
        {"explanation": "first", "parameters": {"Cat": "A", "Num": 1.0}},
        {"explanation": "second", "parameters": {"Num": 2.0}},
    ]
    with pytest.raises(LLMInvalidParameterValueError, match="missing values"):
        parse_llm_response(json.dumps(suggestions), searchspace)


def test_parse_llm_response_continuous_only_happy_path():
    """A valid response for a continuous-only space is parsed correctly."""
    searchspace = SearchSpace.from_product(
        [
            NumericalContinuousParameter("x", bounds=(0.0, 10.0)),
            NumericalContinuousParameter("y", bounds=(0.0, 5.0)),
        ]
    )
    suggestions = [
        {"explanation": "first", "parameters": {"x": 3.0, "y": 2.0}},
        {"explanation": "second", "parameters": {"x": 7.0, "y": 4.0}},
    ]
    result = parse_llm_response(json.dumps(suggestions), searchspace)
    assert len(result) == 2
    assert set(result.columns) == {"x", "y"}
    assert list(result["x"]) == [3.0, 7.0]
    assert list(result["y"]) == [2.0, 4.0]
