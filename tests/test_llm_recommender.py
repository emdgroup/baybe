"""Tests for the LLM-based recommender."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from baybe._optional.info import LLM_INSTALLED
from baybe.exceptions import LLMResponseError
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace import SearchSpace

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
        objective_description="Maximize yield",
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
def test_recommend_with_related_data(mock_completion, searchspace, valid_response):
    """Related data is included in the prompt when provided."""
    from baybe.recommenders.pure.llm.llm import LLMRecommender

    related_data = pd.DataFrame(
        {"temperature": [40.0], "pressure": [2.5], "n_cycles": [3], "yield": [0.9]}
    )
    recommender = LLMRecommender(
        model="gpt-5.4",
        experiment_description="Test",
        objective_description="Maximize yield",
        related_data=related_data,
    )
    mock_completion.return_value = valid_response

    recommender.recommend(batch_size=3, searchspace=searchspace)

    prompt_content = mock_completion.call_args.kwargs.get(
        "messages", mock_completion.call_args[1]["messages"]
    )[0]["content"]
    assert "RELATED DATA" in prompt_content


@patch("baybe._optional.llm.completion")
def test_recommend_with_format_instructions(
    mock_completion, searchspace, valid_response
):
    """Custom format instructions are included in the prompt."""
    from baybe.recommenders.pure.llm.llm import LLMRecommender

    custom_instructions = "Return CSV format instead."
    recommender = LLMRecommender(
        model="gpt-5.4",
        experiment_description="Test",
        objective_description="Maximize yield",
        format_instructions=custom_instructions,
    )
    mock_completion.return_value = valid_response

    recommender.recommend(batch_size=3, searchspace=searchspace)

    prompt_content = mock_completion.call_args.kwargs.get(
        "messages", mock_completion.call_args[1]["messages"]
    )[0]["content"]
    assert custom_instructions in prompt_content


@patch("baybe._optional.llm.completion")
def test_recovery_with_distinct_model(mock_completion, recommender, searchspace):
    """Recovery uses the specified recovery_model and recovery_litellm_args."""
    from baybe.recommenders.pure.llm.llm import LLMRecommender

    recommender = LLMRecommender(
        model="gpt-5.4",
        experiment_description="Test",
        objective_description="Maximize yield",
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
    ],
)
def test_parse_llm_response_errors(
    response_content, error_match, recommender, searchspace
):
    """Malformed responses raise LLMResponseError with descriptive messages."""
    with pytest.raises(LLMResponseError, match=error_match):
        recommender._parse_llm_response(response_content, searchspace)


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
def test_feasibility_filtering(mock_completion, searchspace):
    """Feasibility filtering respects the is_feasible_experiment callback."""
    from baybe.recommenders.pure.llm.llm import LLMRecommender

    recommender = LLMRecommender(
        model="gpt-5.4",
        experiment_description="Test",
        objective_description="Maximize yield",
        is_feasible_experiment=lambda row: row["temperature"] > 20.0,
        overflow_experiments=1,
    )

    mock_completion.return_value = _mock_response(
        _make_suggestions(
            [
                {"temperature": 10.0, "pressure": 1.0, "n_cycles": 1, "catalyst": "A"},
                {"temperature": 50.0, "pressure": 2.0, "n_cycles": 2, "catalyst": "B"},
                {"temperature": 30.0, "pressure": 3.0, "n_cycles": 3, "catalyst": "C"},
                {"temperature": 60.0, "pressure": 4.0, "n_cycles": 4, "catalyst": "A"},
            ]
        )
    )

    recommendations = recommender.recommend(batch_size=3, searchspace=searchspace)

    assert len(recommendations) == 3
    assert all(recommendations["temperature"] > 20.0)


@patch("baybe._optional.llm.completion")
def test_feasibility_filtering_insufficient_feasible(mock_completion, searchspace):
    """Warning is emitted when fewer feasible experiments than batch_size."""
    from baybe.recommenders.pure.llm.llm import LLMRecommender

    recommender = LLMRecommender(
        model="gpt-5.4",
        experiment_description="Test",
        objective_description="Maximize yield",
        is_feasible_experiment=lambda row: row["temperature"] > 90.0,
        overflow_experiments=1,
    )

    mock_completion.return_value = _mock_response(
        _make_suggestions(
            [
                {"temperature": 10.0, "pressure": 1.0, "n_cycles": 1, "catalyst": "A"},
                {"temperature": 50.0, "pressure": 2.0, "n_cycles": 2, "catalyst": "B"},
                {"temperature": 95.0, "pressure": 3.0, "n_cycles": 3, "catalyst": "C"},
                {"temperature": 30.0, "pressure": 4.0, "n_cycles": 4, "catalyst": "A"},
            ]
        )
    )

    with pytest.warns(UserWarning, match="feasibility check"):
        recommendations = recommender.recommend(batch_size=3, searchspace=searchspace)

    assert len(recommendations) == 1
    assert recommendations["temperature"].iloc[0] == 95.0


@patch("baybe._optional.llm.completion")
def test_batch_size_warning_when_llm_returns_fewer(
    mock_completion, recommender, searchspace
):
    """Warning is emitted when LLM returns fewer suggestions than requested."""
    mock_completion.return_value = _mock_response(
        _make_suggestions(
            [
                {"temperature": 25.0, "pressure": 2.0, "n_cycles": 1, "catalyst": "A"},
            ]
        )
    )

    with pytest.warns(UserWarning, match="instead of the requested"):
        recommender.recommend(batch_size=3, searchspace=searchspace)


def test_initialization(recommender):
    """LLMRecommender initializes with correct attributes."""
    assert recommender.model == "gpt-5.4"
    assert recommender.experiment_description == "Test experiment"
    assert recommender.objective_description == "Maximize yield"


def test_initialization_validation():
    """Empty required fields raise during construction."""
    from baybe.recommenders.pure.llm.llm import LLMRecommender

    with pytest.raises(ValueError, match="Length"):
        LLMRecommender(
            model="", experiment_description="desc", objective_description="obj"
        )

    with pytest.raises(ValueError, match="Length"):
        LLMRecommender(
            model="m", experiment_description="", objective_description="obj"
        )

    with pytest.raises(ValueError, match="Length"):
        LLMRecommender(
            model="m", experiment_description="desc", objective_description=""
        )


def test_str_representation(recommender):
    """String representation includes key information."""
    s = str(recommender)
    assert "LLMRecommender" in s
    assert "gpt-5.4" in s
