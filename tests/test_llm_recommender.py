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
from baybe.exceptions import LLMResponseError, LLMResponseWarning
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
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


@patch("baybe._optional.llm.completion")
def test_recovery_with_distinct_model(mock_completion, recommender, searchspace):
    """Recovery uses the specified recovery_model and recovery_litellm_args."""
    from baybe.recommenders.pure.llm.llm import LLMRecommender

    recommender = LLMRecommender(
        model="gpt-5.4",
        experiment_description="Test",
        recovery_model="gpt-4o-mini",
        recovery_litellm_args={"temperature": 0.0},
    )

    invalid = _mock_response("Invalid JSON")
    valid = _mock_response(
        _make_suggestions(
            [
                {"temperature": 50.0, "pressure": 3.0, "n_cycles": 2, "catalyst": "C"},
            ]
        )
    )
    mock_completion.side_effect = [invalid, valid]

    recommender.recommend(batch_size=1, searchspace=searchspace)

    recovery_call = mock_completion.call_args_list[1]
    assert (
        recovery_call.kwargs.get("model", recovery_call[1].get("model"))
        == "gpt-4o-mini"
    )
    assert (
        recovery_call.kwargs.get("temperature", recovery_call[1].get("temperature"))
        == 0.0
    )


@pytest.mark.parametrize(
    ("response_content", "error_match"),
    [
        pytest.param(
            "Invalid JSON",
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
            "outside bounds",
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
            "Invalid values",
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
            "Missing parameter",
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
            "unknown parameter names",
            id="unknown_parameter",
        ),
        pytest.param(
            json.dumps({"explanation": "Test", "parameters": {}}),
            "Response must be a JSON array",
            id="not_a_list",
        ),
        pytest.param(
            json.dumps([]),
            "empty array with no suggestions",
            id="empty_array",
        ),
        pytest.param(
            json.dumps(["a string"]),
            "Each suggestion must be a JSON object",
            id="suggestion_not_dict",
        ),
        pytest.param(
            json.dumps([{"explanation": "Test", "parameters": [1, 2]}]),
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
            "Non-finite or non-numeric values",
            id="non_numeric_continuous",
        ),
    ],
)
def test_parse_llm_response_errors(
    response_content, error_match, recommender, searchspace
):
    """Malformed responses raise LLMResponseError with descriptive messages."""
    with pytest.raises(LLMResponseError, match=error_match):
        recommender._parse_llm_response(response_content, searchspace)


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
    from baybe.recommenders.pure.llm.llm import LLMRecommender

    space = SearchSpace.from_product(parameters=parameters, constraints=constraints)
    rec = LLMRecommender(model="m", experiment_description="test")
    response = _make_suggestions(violation_suggestions)
    with pytest.raises(LLMResponseError, match="violate the.*constraint"):
        rec._parse_llm_response(response, space)


def test_parse_llm_response_rejects_batch_constraint_violation():
    """Batch suggestions with mixed values for a DiscreteBatchConstraint param raise."""
    from baybe.recommenders.pure.llm.llm import LLMRecommender

    parameters = [
        NumericalDiscreteParameter("x", values=[1, 2, 3]),
        CategoricalParameter("y", values=["a", "b"]),
    ]
    space = SearchSpace.from_product(
        parameters=parameters,
        constraints=[DiscreteBatchConstraint(parameters=["x"])],
    )
    rec = LLMRecommender(model="m", experiment_description="test")
    # x values differ across suggestions — violates the batch constraint
    response = _make_suggestions([{"x": 1, "y": "a"}, {"x": 2, "y": "b"}])
    with pytest.raises(LLMResponseError, match="DiscreteBatchConstraint"):
        rec._parse_llm_response(response, space)


def test_parse_llm_response_aligns_index_with_exp_rep():
    """Returned DataFrame index matches the exp_rep index of the search space."""
    from baybe.recommenders.pure.llm.llm import LLMRecommender

    parameters = [
        NumericalDiscreteParameter("x", values=[1, 2, 3]),
        NumericalDiscreteParameter("y", values=[10, 20, 30]),
    ]
    space = SearchSpace.from_product(parameters=parameters)
    rec = LLMRecommender(model="m", experiment_description="test")

    # Suggest the last row of exp_rep — its index is not 0
    last_row = space.discrete.exp_rep.iloc[-1]
    response = _make_suggestions([{"x": last_row["x"], "y": last_row["y"]}])
    result = rec._parse_llm_response(response, space)

    assert list(result.index) == [space.discrete.exp_rep.index[-1]]


def test_parse_llm_response_warns_for_continuous_constraints():
    """A warning is issued when the search space has continuous constraints."""
    from baybe.constraints.continuous import ContinuousLinearConstraint
    from baybe.recommenders.pure.llm.llm import LLMRecommender

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
    rec = LLMRecommender(model="m", experiment_description="test")
    response = _make_suggestions([{"x": 0.3, "y": 0.4}])
    with pytest.warns(LLMResponseWarning, match="continuous constraints"):
        rec._parse_llm_response(response, space)


@pytest.mark.parametrize(
    "wrapper",
    [
        "```json\n{payload}\n```",
        "```\n{payload}\n```",
        "Here are the suggestions:\n{payload}\nHope this helps!",
    ],
    ids=["json_fence", "bare_fence", "surrounding_prose"],
)
def test_parse_llm_response_strips_wrappers(wrapper, recommender, searchspace):
    """Markdown fences and surrounding prose are stripped before JSON parsing."""
    payload = _make_suggestions(
        [{"temperature": 25.0, "pressure": 2.0, "n_cycles": 1, "catalyst": "A"}]
    )
    df = recommender._parse_llm_response(wrapper.format(payload=payload), searchspace)
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

    with pytest.raises(ValueError, match=f"must not contain {phrase}"):
        LLMRecommender(
            model="m",
            experiment_description="desc",
            recovery_litellm_args={key: "secret"},
        )


def test_str_representation(recommender):
    """String representation includes key information."""
    s = str(recommender)
    assert "LLMRecommender" in s
    assert "gpt-5.4" in s
