"""Tests for the LLM-based recommender."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from baybe._optional.info import LLM_INSTALLED
from baybe.constraints.conditions import SubSelectionCondition, ThresholdCondition
from baybe.constraints.discrete import (
    DiscreteBatchConstraint,
    DiscreteCardinalityConstraint,
    DiscreteConstraint,
    DiscreteCustomConstraint,
    DiscreteDependenciesConstraint,
    DiscreteExcludeConstraint,
    DiscreteLinkedParametersConstraint,
    DiscreteNoLabelDuplicatesConstraint,
    DiscretePermutationInvarianceConstraint,
    DiscreteProductConstraint,
    DiscreteSumConstraint,
)
from baybe.exceptions import (
    ConstraintViolationError,
    IneligiblePointsError,
    InvalidParameterValueError,
    LLMResponseError,
    LLMResponseWarning,
    MalformedLLMResponseError,
    MissingParameterError,
    NonNumericParameterError,
    UnknownParameterError,
)
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.recommenders.pure.llm._parsing import parse_llm_response
from baybe.searchspace import SearchSpace
from baybe.utils.basic import get_subclasses

pytestmark = pytest.mark.skipif(
    not LLM_INSTALLED, reason="LLM dependencies not installed"
)


def _mock_response(content: str) -> SimpleNamespace:
    """Create a mock LLM response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _make_suggestions(params_list: list[dict]) -> str:
    """Create a JSON string of suggestions from a list of parameter dicts."""
    return json.dumps(
        [
            {"explanation": f"Suggestion {i}", "parameters": p}
            for i, p in enumerate(params_list)
        ]
    )


def _filtered_discrete_space(exclude: dict):
    """Build a 3x3 discrete space with one candidate filtered from the eligible set."""
    from attrs import evolve

    from baybe.searchspace._filtered import FilteredSubspaceDiscrete

    space = SearchSpace.from_product(
        [
            NumericalDiscreteParameter("x", values=[1, 2, 3]),
            NumericalDiscreteParameter("y", values=[1, 2, 3]),
        ]
    )
    exp_rep = space.discrete.exp_rep
    mask_keep = ~(
        (exp_rep["x"] == exclude["x"]) & (exp_rep["y"] == exclude["y"])
    ).to_numpy()
    return evolve(
        space,
        discrete=FilteredSubspaceDiscrete.from_subspace(space.discrete, mask_keep),
    )


@pytest.fixture(name="searchspace")
def fixture_searchspace():
    """A search space with continuous, discrete numeric, and categorical parameters."""
    parameters = [
        NumericalContinuousParameter(
            name="temperature",
            bounds=(0.0, 100.0),
            metadata={"description": "Reaction temperature", "unit": "°C"},
        ),
        NumericalContinuousParameter(
            name="pressure",
            bounds=(0.0, 5.0),
            metadata={"description": "Reaction pressure", "unit": "bar"},
        ),
        NumericalDiscreteParameter(
            name="n_cycles",
            values=[1, 2, 3, 4, 5],
            metadata={"description": "Number of reaction cycles"},
        ),
        CategoricalParameter(
            name="catalyst",
            values=["A", "B", "C"],
            metadata={"description": "Catalyst type"},
        ),
    ]
    return SearchSpace.from_product(parameters)


@pytest.fixture(name="recommender")
def fixture_recommender():
    """An LLMRecommender instance for testing."""
    from baybe.recommenders.pure.llm.llm import LLMRecommender

    return LLMRecommender(
        model="gpt-5.4",
        experiment_description="Test experiment",
    )


@pytest.fixture(name="valid_response")
def fixture_valid_response():
    """A valid mock LLM response with three suggestions."""
    return _mock_response(
        _make_suggestions(
            [
                {"temperature": 25.0, "pressure": 2.0, "n_cycles": 1, "catalyst": "A"},
                {"temperature": 30.0, "pressure": 1.5, "n_cycles": 3, "catalyst": "B"},
                {"temperature": 50.0, "pressure": 3.0, "n_cycles": 5, "catalyst": "C"},
            ]
        )
    )


@patch("baybe._optional.llm.completion")
def test_recommend_success(mock_completion, recommender, searchspace, valid_response):
    """Successful recommendation returns a DataFrame with correct shape."""
    mock_completion.return_value = valid_response

    recommendations = recommender.recommend(batch_size=3, searchspace=searchspace)

    assert isinstance(recommendations, pd.DataFrame)
    assert len(recommendations) == 3
    assert set(recommendations.columns) == {
        "temperature",
        "pressure",
        "n_cycles",
        "catalyst",
    }
    assert recommendations["temperature"].tolist() == [25.0, 30.0, 50.0]
    assert recommendations["catalyst"].tolist() == ["A", "B", "C"]
    assert recommendations["n_cycles"].tolist() == [1, 3, 5]


@patch("baybe._optional.llm.completion")
def test_recommend_with_measurements(
    mock_completion, recommender, searchspace, valid_response
):
    """Recommendations include previous measurements in prompt."""
    mock_completion.return_value = valid_response

    measurements = pd.DataFrame(
        {
            "temperature": [20.0, 25.0],
            "pressure": [1.0, 2.0],
            "n_cycles": [1, 2],
            "catalyst": ["A", "B"],
            "yield": [0.5, 0.7],
        }
    )

    recommendations = recommender.recommend(
        batch_size=3, searchspace=searchspace, measurements=measurements
    )

    assert isinstance(recommendations, pd.DataFrame)
    assert len(recommendations) == 3
    prompt_content = mock_completion.call_args.kwargs.get(
        "messages", mock_completion.call_args[1]["messages"]
    )[0]["content"]
    assert "PREVIOUS MEASUREMENTS" in prompt_content


@patch("baybe._optional.llm.completion")
def test_recommend_with_pending_experiments(
    mock_completion, recommender, searchspace, valid_response
):
    """Pending experiments are accepted and included in the prompt."""
    mock_completion.return_value = valid_response

    pending_experiments = pd.DataFrame(
        {
            "temperature": [10.0, 15.0],
            "pressure": [1.0, 2.0],
            "n_cycles": [2, 4],
            "catalyst": ["A", "C"],
        }
    )

    recommendations = recommender.recommend(
        batch_size=3,
        searchspace=searchspace,
        pending_experiments=pending_experiments,
    )

    assert isinstance(recommendations, pd.DataFrame)
    assert len(recommendations) == 3
    prompt_content = mock_completion.call_args.kwargs.get(
        "messages", mock_completion.call_args[1]["messages"]
    )[0]["content"]
    assert "PENDING EXPERIMENTS" in prompt_content


@patch("baybe._optional.llm.completion")
def test_recommend_with_objective(
    mock_completion, recommender, searchspace, valid_response
):
    """A passed objective is rendered into the prompt."""
    from baybe.objectives.single import SingleTargetObjective
    from baybe.targets.numerical import NumericalTarget

    mock_completion.return_value = valid_response
    objective = SingleTargetObjective(NumericalTarget("yield", minimize=False))

    with pytest.warns(UserWarning, match="objective has no metadata description"):
        recommender.recommend(
            batch_size=3,
            searchspace=searchspace,
            objective=objective,
        )

    prompt_content = mock_completion.call_args.kwargs.get(
        "messages", mock_completion.call_args[1]["messages"]
    )[0]["content"]
    assert "OPTIMIZATION TARGETS" in prompt_content
    assert "yield" in prompt_content


@pytest.mark.parametrize(
    ("response_content", "error_type", "error_match"),
    [
        pytest.param(
            "Invalid JSON",
            MalformedLLMResponseError,
            "Error parsing JSON output",
            id="invalid_json",
        ),
        pytest.param(
            json.dumps(
                [
                    {
                        "explanation": "Test",
                        "parameters": {
                            "temperature": 150.0,
                            "pressure": 2.0,
                            "n_cycles": 1,
                            "catalyst": "A",
                        },
                    }
                ]
            ),
            InvalidParameterValueError,
            "has invalid values in parameter",
            id="out_of_bounds",
        ),
        pytest.param(
            json.dumps(
                [
                    {
                        "explanation": "Test",
                        "parameters": {
                            "temperature": 25.0,
                            "pressure": 2.0,
                            "n_cycles": 1,
                            "catalyst": "D",
                        },
                    }
                ]
            ),
            InvalidParameterValueError,
            "has invalid values in parameter",
            id="invalid_categorical",
        ),
        pytest.param(
            json.dumps(
                [
                    {
                        "explanation": "Test",
                        "parameters": {"temperature": 25.0, "catalyst": "A"},
                    }
                ]
            ),
            MissingParameterError,
            "missing values for the following parameters",
            id="missing_parameter",
        ),
        pytest.param(
            json.dumps(
                [
                    {
                        "explanation": "Test",
                        "parameters": {
                            "temperature": 25.0,
                            "pressure": 2.0,
                            "n_cycles": 1,
                            "catalyst": "A",
                            "unknown": 1,
                        },
                    }
                ]
            ),
            UnknownParameterError,
            "unknown parameter names",
            id="unknown_parameter",
        ),
        pytest.param(
            json.dumps({"explanation": "Test", "parameters": {}}),
            MalformedLLMResponseError,
            "Response must be a JSON array",
            id="not_a_list",
        ),
        pytest.param(
            json.dumps([]),
            MalformedLLMResponseError,
            "empty array with no suggestions",
            id="empty_array",
        ),
        pytest.param(
            json.dumps(["a string"]),
            MalformedLLMResponseError,
            "Each suggestion must be a JSON object",
            id="suggestion_not_dict",
        ),
        pytest.param(
            json.dumps([{"explanation": "Test", "parameters": [1, 2]}]),
            MalformedLLMResponseError,
            "Parameters must be a JSON object",
            id="parameters_not_dict",
        ),
        pytest.param(
            json.dumps(
                [
                    {
                        "parameters": {
                            "temperature": 25.0,
                            "pressure": 2.0,
                            "n_cycles": 1,
                            "catalyst": "A",
                        }
                    }
                ]
            ),
            MalformedLLMResponseError,
            "must contain an 'explanation' field",
            id="missing_explanation",
        ),
        pytest.param(
            json.dumps(
                [
                    {
                        "explanation": "Test",
                        "parameters": {
                            "temperature": "hot",
                            "pressure": 2.0,
                            "n_cycles": 1,
                            "catalyst": "A",
                        },
                    }
                ]
            ),
            NonNumericParameterError,
            "has non-numeric entries",
            id="non_numeric_continuous",
        ),
    ],
)
def test_parse_llm_response_errors(
    response_content, error_type, error_match, searchspace
):
    """Malformed responses raise the specific error subtype with a clear message."""
    with pytest.raises(error_type, match=error_match):
        parse_llm_response(response_content, searchspace)


def test_parse_llm_response_numerical_tolerance_snaps_to_nearest():
    """A numerical-discrete value within tolerance is accepted and snapped.

    Matches how user measurement input is handled: ``validate_parameter_input``
    accepts values within the parameter tolerance and ``fuzzy_row_match`` snaps
    them to the nearest allowed value.
    """
    space = SearchSpace.from_product(
        [NumericalDiscreteParameter("x", values=[1.0, 2.0, 3.0], tolerance=0.4)]
    )
    # 1.3 is within tolerance 0.4 of 1.0 but not an exact allowed value.
    result = parse_llm_response(_make_suggestions([{"x": 1.3}]), space)
    assert result["x"].tolist() == [1.0]


def test_parse_llm_response_numerical_out_of_tolerance_rejected():
    """A numerical-discrete value outside tolerance is rejected."""
    space = SearchSpace.from_product(
        [NumericalDiscreteParameter("x", values=[1.0, 2.0, 3.0], tolerance=0.1)]
    )
    # 1.3 is outside tolerance 0.1 of every allowed value.
    with pytest.raises(LLMResponseError, match="has invalid values in parameter"):
        parse_llm_response(_make_suggestions([{"x": 1.3}]), space)


# ---------------------------------------------------------------------------
# Constraint violation test cases — one per concrete DiscreteConstraint class
# (except DiscreteBatchConstraint, which is tested separately below).
#
# Each entry maps a constraint class to (parameters, constraints, violations)
# where violations is a list of suggestion dicts that individually satisfy all
# parameter bounds/values but violate the constraint as a combination or batch.
# ---------------------------------------------------------------------------
_ROW_CONSTRAINT_VIOLATION_CASES = {
    DiscreteExcludeConstraint: (
        [
            NumericalDiscreteParameter("x", values=[1, 2, 3]),
            CategoricalParameter("y", values=["a", "b", "c"]),
        ],
        [
            DiscreteExcludeConstraint(
                parameters=["x", "y"],
                conditions=[
                    SubSelectionCondition(selection=[2]),
                    SubSelectionCondition(selection=["b"]),
                ],
                combiner="AND",
            )
        ],
        [{"x": 2, "y": "b"}],
    ),
    DiscreteSumConstraint: (
        [
            NumericalDiscreteParameter("a", values=[1, 2, 3]),
            NumericalDiscreteParameter("b", values=[1, 2, 3]),
        ],
        [
            DiscreteSumConstraint(
                parameters=["a", "b"],
                condition=ThresholdCondition(threshold=6.0, operator="="),
            )
        ],
        [{"a": 1, "b": 1}],  # sum=2, required sum=6
    ),
    DiscreteProductConstraint: (
        [
            NumericalDiscreteParameter("a", values=[1, 2, 3]),
            NumericalDiscreteParameter("b", values=[1, 2, 3]),
        ],
        [
            DiscreteProductConstraint(
                parameters=["a", "b"],
                condition=ThresholdCondition(threshold=6.0, operator=">="),
            )
        ],
        [{"a": 1, "b": 1}],  # product=1, required product>=6
    ),
    DiscreteNoLabelDuplicatesConstraint: (
        [
            CategoricalParameter("x", values=["A", "B", "C"]),
            CategoricalParameter("y", values=["A", "B", "C"]),
        ],
        [DiscreteNoLabelDuplicatesConstraint(parameters=["x", "y"])],
        [{"x": "A", "y": "A"}],  # duplicate label across parameters
    ),
    DiscreteLinkedParametersConstraint: (
        [
            NumericalDiscreteParameter("x", values=[1, 2, 3]),
            NumericalDiscreteParameter("y", values=[1, 2, 3]),
        ],
        [DiscreteLinkedParametersConstraint(parameters=["x", "y"])],
        [{"x": 1, "y": 2}],  # x and y must be equal
    ),
    DiscreteDependenciesConstraint: (
        [
            CategoricalParameter("switch", values=["on", "off"]),
            CategoricalParameter("mode", values=["fast", "slow"]),
        ],
        [
            DiscreteDependenciesConstraint(
                parameters=["switch"],
                conditions=[SubSelectionCondition(selection=["on"])],
                affected_parameters=[["mode"]],
            )
        ],
        # when switch="off" mode is irrelevant: two "off" rows are duplicates
        [{"switch": "off", "mode": "fast"}, {"switch": "off", "mode": "slow"}],
    ),
    DiscretePermutationInvarianceConstraint: (
        [
            NumericalDiscreteParameter("a", values=[1, 2, 3]),
            NumericalDiscreteParameter("b", values=[1, 2, 3]),
        ],
        [DiscretePermutationInvarianceConstraint(parameters=["a", "b"])],
        # (1,2) and (2,1) are permutation-equivalent; the second is a duplicate
        [{"a": 1, "b": 2}, {"a": 2, "b": 1}],
    ),
    DiscreteCustomConstraint: (
        [NumericalDiscreteParameter("x", values=[1, 2, 3])],
        [DiscreteCustomConstraint(parameters=["x"], validator=lambda df: df["x"] != 2)],
        [{"x": 2}],  # custom validator rejects x=2
    ),
    DiscreteCardinalityConstraint: (
        [
            NumericalDiscreteParameter("a", values=[0, 1, 2]),
            NumericalDiscreteParameter("b", values=[0, 1, 2]),
        ],
        [DiscreteCardinalityConstraint(parameters=["a", "b"], max_cardinality=1)],
        [{"a": 1, "b": 1}],  # 2 nonzero values exceeds max_cardinality=1
    ),
}

# Verify all concrete DiscreteConstraint subclasses (except DiscreteBatchConstraint)
# have a violation test case — fails at collection time if coverage lapses.
_ALL_ROW_CONSTRAINT_CLASSES = frozenset(
    cls
    for cls in get_subclasses(DiscreteConstraint)
    if cls is not DiscreteBatchConstraint
)
assert frozenset(_ROW_CONSTRAINT_VIOLATION_CASES) == _ALL_ROW_CONSTRAINT_CLASSES, (
    "Missing constraint violation cases for: "
    f"{_ALL_ROW_CONSTRAINT_CLASSES - frozenset(_ROW_CONSTRAINT_VIOLATION_CASES)}"
)


@pytest.mark.parametrize(
    ("parameters", "constraints", "violation_suggestions"),
    [
        pytest.param(*_ROW_CONSTRAINT_VIOLATION_CASES[cls], id=cls.__name__)
        for cls in get_subclasses(DiscreteConstraint)
        if cls is not DiscreteBatchConstraint
    ],
)
def test_parse_llm_response_rejects_row_constraint_violations(
    parameters, constraints, violation_suggestions
):
    """Suggestions valid per-parameter but violating a discrete constraint raise."""
    space = SearchSpace.from_product(parameters=parameters, constraints=constraints)
    response = _make_suggestions(violation_suggestions)
    with pytest.raises(ConstraintViolationError, match="violate the.*constraint"):
        parse_llm_response(response, space)


def test_parse_llm_response_rejects_batch_constraint_violation():
    """Batch suggestions with mixed values for a DiscreteBatchConstraint param raise."""
    parameters = [
        NumericalDiscreteParameter("x", values=[1, 2, 3]),
        CategoricalParameter("y", values=["a", "b"]),
    ]
    space = SearchSpace.from_product(
        parameters=parameters,
        constraints=[DiscreteBatchConstraint(parameters=["x"])],
    )
    # x values differ across suggestions — violates the batch constraint
    response = _make_suggestions([{"x": 1, "y": "a"}, {"x": 2, "y": "b"}])
    with pytest.raises(ConstraintViolationError, match="DiscreteBatchConstraint"):
        parse_llm_response(response, space)


def test_parse_llm_response_aligns_index_with_exp_rep():
    """Returned DataFrame index matches the exp_rep index of the search space."""
    parameters = [
        NumericalDiscreteParameter("x", values=[1, 2, 3]),
        NumericalDiscreteParameter("y", values=[10, 20, 30]),
    ]
    space = SearchSpace.from_product(parameters=parameters)
    # Suggest the last row of exp_rep — its index is not 0
    last_row = space.discrete.exp_rep.iloc[-1]
    response = _make_suggestions([{"x": last_row["x"], "y": last_row["y"]}])
    result = parse_llm_response(response, space)

    assert list(result.index) == [space.discrete.exp_rep.index[-1]]


def test_parse_llm_response_warns_for_continuous_constraints():
    """A warning is issued when the search space has continuous constraints."""
    from baybe.constraints.continuous import ContinuousLinearConstraint

    parameters = [
        NumericalContinuousParameter("x", bounds=(0, 1)),
        NumericalContinuousParameter("y", bounds=(0, 1)),
    ]
    space = SearchSpace.from_product(
        parameters=parameters,
        constraints=[
            ContinuousLinearConstraint(
                parameters=["x", "y"], coefficients=[1, 1], rhs=1.5, operator="<="
            )
        ],
    )
    response = _make_suggestions([{"x": 0.3, "y": 0.4}])
    with pytest.warns(LLMResponseWarning, match="continuous constraints"):
        parse_llm_response(response, space)


@pytest.mark.parametrize(
    "wrapper",
    [
        "```json\n{payload}\n```",
        "```\n{payload}\n```",
        "Here are the suggestions:\n{payload}\nHope this helps!",
    ],
    ids=["json_fence", "bare_fence", "surrounding_prose"],
)
def test_parse_llm_response_strips_wrappers(wrapper, searchspace):
    """Markdown fences and surrounding prose are stripped before JSON parsing."""
    payload = _make_suggestions(
        [{"temperature": 25.0, "pressure": 2.0, "n_cycles": 1, "catalyst": "A"}]
    )
    df = parse_llm_response(wrapper.format(payload=payload), searchspace)
    assert len(df) == 1
    assert df["catalyst"].iloc[0] == "A"


@patch("baybe._optional.llm.completion")
def test_recommend_invalid_response_with_failed_recovery(
    mock_completion, recommender, searchspace
):
    """Invalid response that also fails recovery raises LLMResponseError."""
    mock_completion.return_value = _mock_response("Invalid JSON")

    with pytest.raises(LLMResponseError, match="Recovery produced another malformed"):
        recommender.recommend(batch_size=3, searchspace=searchspace)


@patch("baybe._optional.llm.completion")
def test_recovery_success(mock_completion, recommender, searchspace):
    """Successful recovery from a malformed initial response."""
    invalid = _mock_response("Invalid JSON")
    valid = _mock_response(
        _make_suggestions(
            [
                {"temperature": 50.0, "pressure": 3.0, "n_cycles": 2, "catalyst": "C"},
            ]
        )
    )
    mock_completion.side_effect = [invalid, valid]

    recommendations = recommender.recommend(batch_size=1, searchspace=searchspace)

    assert len(recommendations) == 1
    assert mock_completion.call_count == 2


@patch("baybe._optional.llm.completion")
def test_batch_size_error_when_llm_returns_fewer(
    mock_completion, recommender, searchspace
):
    """An error is raised when LLM returns fewer suggestions than requested."""
    mock_completion.return_value = _mock_response(
        _make_suggestions(
            [
                {"temperature": 25.0, "pressure": 2.0, "n_cycles": 1, "catalyst": "A"},
            ]
        )
    )

    with pytest.raises(LLMResponseError, match="instead of the requested"):
        recommender.recommend(batch_size=3, searchspace=searchspace)


@patch("baybe._optional.llm.completion")
def test_completion_failure_wrapped(mock_completion, recommender, searchspace):
    """A failing completion call is wrapped in an LLMResponseError."""
    mock_completion.side_effect = RuntimeError("boom")

    with pytest.raises(LLMResponseError, match="call to the language model failed"):
        recommender.recommend(batch_size=3, searchspace=searchspace)


def test_recommend_rejects_nonpositive_batch_size(recommender, searchspace):
    """A batch size below 1 raises ValueError before any model call."""
    with pytest.raises(ValueError, match="at least request one recommendation"):
        recommender.recommend(batch_size=0, searchspace=searchspace)


@patch("baybe._optional.llm.completion")
def test_recommend_allows_duplicate_configurations(
    mock_completion, recommender, searchspace
):
    """Duplicate configurations within a batch are permitted."""
    dup = {"temperature": 25.0, "pressure": 2.0, "n_cycles": 1, "catalyst": "A"}
    mock_completion.return_value = _mock_response(_make_suggestions([dup, dup, dup]))

    recommendations = recommender.recommend(batch_size=3, searchspace=searchspace)

    assert len(recommendations) == 3
    assert recommendations["catalyst"].tolist() == ["A", "A", "A"]


def test_parse_llm_response_rejects_ineligible_points():
    """Suggestions matching filtered-out (ineligible) candidates raise."""
    filtered_space = _filtered_discrete_space(exclude={"x": 1, "y": 1})
    # The first suggestion targets the excluded candidate; the second is eligible.
    response = _make_suggestions([{"x": 1, "y": 1}, {"x": 2, "y": 2}])
    with pytest.raises(IneligiblePointsError, match="do not correspond to eligible"):
        parse_llm_response(response, filtered_space)


def test_initialization(recommender):
    """LLMRecommender initializes with correct attributes."""
    assert recommender.model == "gpt-5.4"
    assert recommender.experiment_description == "Test experiment"


def test_initialization_validation():
    """Empty required fields raise during construction."""
    from baybe.recommenders.pure.llm.llm import LLMRecommender

    with pytest.raises(ValueError, match="Length"):
        LLMRecommender(model="", experiment_description="desc")

    with pytest.raises(ValueError, match="Length"):
        LLMRecommender(model="m", experiment_description="")


@pytest.mark.parametrize(
    ("key", "phrase"),
    [
        ("api_key", "credential keys"),
        ("api_base", "credential keys"),
        ("api_version", "credential keys"),
        ("model", "keys that are set"),
        ("messages", "keys that are set"),
    ],
    ids=["api_key", "api_base", "api_version", "model", "messages"],
)
def test_construction_rejects_protected_keys(key, phrase):
    """Credential or preset keys raise during construction."""
    from baybe.recommenders.pure.llm.llm import LLMRecommender

    with pytest.raises(ValueError, match=f"must not contain {phrase}"):
        LLMRecommender(
            model="m",
            experiment_description="desc",
            litellm_args={key: "secret"},
        )


def test_str_representation(recommender):
    """String representation includes key information."""
    s = str(recommender)
    assert "LLMRecommender" in s
    assert "gpt-5.4" in s
