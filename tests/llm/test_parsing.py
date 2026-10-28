"""Tests for LLM response parsing and validation."""

import json

import pytest

from baybe.constraints.continuous import ContinuousLinearConstraint
from baybe.constraints.discrete import DiscreteBatchConstraint
from baybe.exceptions import (
    LLMBatchConstraintValidationError,
    LLMBatchSizeError,
    LLMDuplicateRecommendationsError,
    LLMIneligiblePointsError,
    LLMMalformedResponseError,
    LLMResponseWarning,
    LLMUnknownParameterError,
)
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.recommenders.pure.llm._parsing import _extract_json_array, parse_llm_response
from baybe.recommenders.pure.llm._schema import (
    _EXPLANATION_FIELD,
    _PARAMETERS_FIELD,
    _response_format,
)
from baybe.searchspace import SearchSpace
from tests.llm._mock import (
    make_discrete_searchspace,
    make_response_json,
)


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
    result = _extract_json_array(response)
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
def test_extract_json_array_raises_on_invalid(response):
    """The extractor raises ``ValueError`` when no array of objects is found."""
    with pytest.raises(ValueError, match="No JSON array"):
        _extract_json_array(response)


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
            "could not be parsed as a JSON array",
            id="no_json",
        ),
        pytest.param(
            "[]",
            "could not be parsed as a JSON array",
            id="empty_array",
        ),
        pytest.param(
            '[{"explanation": "x"}]',
            "does not match the expected format",
            id="no_parameters",
        ),
        pytest.param(
            '[{"parameters": {"any": 1}}]',
            "does not match the expected format",
            id="no_explanation",
        ),
        pytest.param(
            '[{"explanation": "x", "parameters": [1, 2]}]',
            "does not match the expected format",
            id="params_not_dict",
        ),
    ],
)
def test_parse_llm_response_format_errors(discrete_searchspace, response, match):
    """Malformed responses raise ``LLMMalformedResponseError``."""
    with pytest.raises(LLMMalformedResponseError, match=match):
        parse_llm_response(response, discrete_searchspace, batch_size=1)


def test_parse_llm_response_unknown_parameter():
    """Referencing an unknown parameter raises ``LLMUnknownParameterError``."""
    searchspace = SearchSpace.from_product([CategoricalParameter("Cat", ("A", "B"))])
    response = make_response_json(searchspace, Unknown="x")
    with pytest.raises(
        LLMUnknownParameterError, match="Response contains unknown parameter names"
    ):
        parse_llm_response(response, searchspace, batch_size=1)


_CAT_NUM_PARAMETERS = [
    CategoricalParameter("Cat", ("A", "B")),
    NumericalDiscreteParameter("Num", (1.0, 2.0)),
]


@pytest.mark.parametrize(
    ("parameters", "configurations", "match"),
    [
        pytest.param(
            [CategoricalParameter("Cat", ("A", "B"))],
            [{"Cat": "A"}, {"Cat": "Z"}],
            r"^1 configuration\(s\) do not belong to the discrete subspace:\n- Cat=Z:",
            id="valid_then_invalid",
        ),
        pytest.param(
            _CAT_NUM_PARAMETERS,
            [{"Cat": "A", "Num": 1.0}, {"Num": 2.0}],
            r"- Num=2.0: missing values for parameters \['Cat'\]",
            id="partially_missing",
        ),
        pytest.param(
            _CAT_NUM_PARAMETERS,
            [{"Num": 1.0}],
            r"- Num=1.0: missing values for parameters \['Cat'\]",
            id="fully_missing",
        ),
    ],
)
def test_parse_llm_response_ineligible(parameters, configurations, match):
    """Configurations rejected by the search space raise ``LLMIneligiblePointsError``.

    The individual rejection reasons are tested with the search space validation.
    """
    searchspace = SearchSpace.from_product(parameters)
    response = json.dumps(
        [{"explanation": "x", "parameters": c} for c in configurations]
    )
    with pytest.raises(LLMIneligiblePointsError, match=match):
        parse_llm_response(response, searchspace, batch_size=len(configurations))


def test_parse_llm_response_ineligible_recovery_instruction():
    """The recovery instruction forwards the rejection reasons to the model."""
    searchspace = SearchSpace.from_product([CategoricalParameter("Cat", ("A", "B"))])
    response = json.dumps([{"explanation": "x", "parameters": {"Cat": "Z"}}])
    with pytest.raises(LLMIneligiblePointsError, match="Cat=Z") as exc_info:
        parse_llm_response(response, searchspace, batch_size=1)
    instruction = exc_info.value.recovery_instruction
    assert "- Cat=Z: invalid values for parameters ['Cat']" in instruction
    assert instruction.endswith(
        "Propose different configurations that belong to the search space instead."
    )


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
    recommendations = [
        {"explanation": "first", "parameters": {"Cat": "A", "Num": 1.0}},
        {"explanation": "second", "parameters": {"Cat": "B", "Num": 2.0}},
    ]
    response = json.dumps(recommendations)
    with pytest.raises(
        LLMBatchConstraintValidationError, match="DiscreteBatchConstraint"
    ):
        parse_llm_response(response, searchspace, batch_size=2)


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
        parse_llm_response(response, searchspace, batch_size=1)


def test_parse_llm_response_happy_path():
    """A valid response is parsed into a dataframe with correct shape and values."""
    searchspace = SearchSpace.from_product(
        [
            CategoricalParameter("Cat", ("A", "B", "C")),
            NumericalDiscreteParameter("Num", (1.0, 2.0, 3.0)),
        ]
    )
    recommendations = [
        {"explanation": "first", "parameters": {"Cat": "A", "Num": 1.0}},
        {"explanation": "second", "parameters": {"Cat": "B", "Num": 2.0}},
    ]
    result = parse_llm_response(json.dumps(recommendations), searchspace, batch_size=2)
    assert set(result.columns) == {"Cat", "Num"}
    assert list(result["Cat"]) == ["A", "B"]
    assert list(result["Num"]) == [1.0, 2.0]


def test_parse_llm_response_hybrid_output():
    """Hybrid spaces produce output with aligned discrete and continuous parts."""
    searchspace = SearchSpace.from_product(
        [
            CategoricalParameter("Cat", ("A", "B", "C")),
            NumericalContinuousParameter("Cont", bounds=(0.0, 1.0)),
        ]
    )
    recommendations = [
        {"explanation": "first", "parameters": {"Cat": "A", "Cont": 0.3}},
        {"explanation": "second", "parameters": {"Cat": "A", "Cont": 0.7}},
    ]
    result = parse_llm_response(json.dumps(recommendations), searchspace, batch_size=2)
    assert set(result.columns) == {"Cat", "Cont"}
    assert list(result["Cat"]) == ["A", "A"]
    assert list(result["Cont"]) == [0.3, 0.7]


def test_parse_llm_response_continuous_only_happy_path():
    """A valid response for a continuous-only space is parsed correctly."""
    searchspace = SearchSpace.from_product(
        [
            NumericalContinuousParameter("x", bounds=(0.0, 10.0)),
            NumericalContinuousParameter("y", bounds=(0.0, 5.0)),
        ]
    )
    recommendations = [
        {"explanation": "first", "parameters": {"x": 3.0, "y": 2.0}},
        {"explanation": "second", "parameters": {"x": 7.0, "y": 4.0}},
    ]
    result = parse_llm_response(json.dumps(recommendations), searchspace, batch_size=2)
    assert len(result) == 2
    assert set(result.columns) == {"x", "y"}
    assert list(result["x"]) == [3.0, 7.0]
    assert list(result["y"]) == [2.0, 4.0]


@pytest.mark.parametrize(
    ("batch_size", "constraints", "configurations"),
    [
        pytest.param(
            2,
            [],
            [{"Cat": "Z", "Num": 1.0}],
            id="too_few_with_invalid_values",
        ),
        pytest.param(
            1,
            [],
            [{"Cat": "A", "Num": 1.0}, {"Cat": "B", "Num": 2.0}],
            id="valid_surplus",
        ),
        pytest.param(
            1,
            [],
            [{"Cat": "A", "Num": 1.0}, {"Cat": "Z", "Num": 1.0}],
            id="invalid_surplus",
        ),
        pytest.param(
            1,
            [DiscreteBatchConstraint(parameters=["Cat"])],
            [{"Cat": "A", "Num": 1.0}, {"Cat": "B", "Num": 2.0}],
            id="batch_constraint_violated_by_surplus",
        ),
    ],
)
def test_parse_llm_response_wrong_batch_size(batch_size, constraints, configurations):
    """Wrong batch sizes raise before individual or batch-level content validation."""
    searchspace = SearchSpace.from_product(_CAT_NUM_PARAMETERS, constraints)
    recommendations = [{"explanation": "x", "parameters": c} for c in configurations]
    received = len(recommendations)
    with pytest.raises(
        LLMBatchSizeError, match=f"returned {received} recommendation"
    ) as exc_info:
        parse_llm_response(
            json.dumps(recommendations), searchspace, batch_size=batch_size
        )
    error = exc_info.value
    assert error.requested == batch_size
    assert error.received == received
    assert f"provided {received} recommendation(s)" in error.recovery_instruction
    assert f"but exactly {batch_size} are required" in error.recovery_instruction


@pytest.mark.parametrize(
    ("parameter", "values"),
    [
        pytest.param(CategoricalParameter("p", ("A", "B")), ["A", "A"], id="identical"),
        pytest.param(
            NumericalContinuousParameter("p", (0.0, 1.0)),
            [0.5, 0.5],
            id="continuous",
        ),
    ],
)
def test_parse_llm_response_duplicates(parameter, values):
    """Duplicate configurations within a batch raise an error."""
    searchspace = SearchSpace.from_product([parameter])
    recommendations = [{"explanation": "x", "parameters": {"p": v}} for v in values]
    with pytest.raises(LLMDuplicateRecommendationsError, match="1 recommendation"):
        parse_llm_response(json.dumps(recommendations), searchspace, batch_size=2)
