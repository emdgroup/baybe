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


@pytest.fixture(name="searchspace")
def fixture_searchspace():
    """A search space with continuous and categorical parameters."""
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
        model="gpt-3.5-turbo",
        experiment_description="Test experiment",
        objective_description="Maximize yield",
    )


@pytest.fixture(name="valid_response")
def fixture_valid_response():
    """A valid mock LLM response with two suggestions."""
    return _mock_response(
        json.dumps(
            [
                {
                    "explanation": "Test suggestion 1",
                    "parameters": {
                        "temperature": 25.0,
                        "pressure": 2.0,
                        "catalyst": "A",
                    },
                },
                {
                    "explanation": "Test suggestion 2",
                    "parameters": {
                        "temperature": 30.0,
                        "pressure": 1.5,
                        "catalyst": "B",
                    },
                },
            ]
        )
    )


@patch("baybe._optional.llm.completion")
def test_recommend_success(mock_completion, recommender, searchspace, valid_response):
    """Successful recommendation returns a DataFrame with correct shape."""
    mock_completion.return_value = valid_response

    recommendations = recommender.recommend(batch_size=2, searchspace=searchspace)

    assert isinstance(recommendations, pd.DataFrame)
    assert len(recommendations) == 2
    assert set(recommendations.columns) == {"temperature", "pressure", "catalyst"}
    assert recommendations["temperature"].tolist() == [25.0, 30.0]
    assert recommendations["pressure"].tolist() == [2.0, 1.5]
    assert recommendations["catalyst"].tolist() == ["A", "B"]


@patch("baybe._optional.llm.completion")
def test_recommend_with_measurements(
    mock_completion, recommender, searchspace, valid_response
):
    """Recommendations can be generated with previous measurements."""
    mock_completion.return_value = valid_response

    measurements = pd.DataFrame(
        {
            "temperature": [20.0, 25.0],
            "pressure": [1.0, 2.0],
            "catalyst": ["A", "B"],
            "yield": [0.5, 0.7],
        }
    )

    recommendations = recommender.recommend(
        batch_size=2, searchspace=searchspace, measurements=measurements
    )

    assert isinstance(recommendations, pd.DataFrame)
    assert len(recommendations) == 2
    prompt_args = mock_completion.call_args
    assert (
        "PREVIOUS MEASUREMENTS"
        in prompt_args.kwargs.get("messages", prompt_args[1]["messages"])[0]["content"]
    )


@patch("baybe._optional.llm.completion")
def test_recommend_invalid_json(mock_completion, recommender, searchspace):
    """Invalid JSON response raises LLMResponseError after failed recovery."""
    mock_completion.return_value = _mock_response("Invalid JSON")

    with pytest.raises(LLMResponseError, match="Failed to recover"):
        recommender.recommend(batch_size=2, searchspace=searchspace)


@patch("baybe._optional.llm.completion")
def test_recommend_out_of_bounds(mock_completion, recommender, searchspace):
    """Out-of-bounds values raise LLMResponseError after failed recovery."""
    mock_completion.return_value = _mock_response(
        json.dumps(
            [
                {
                    "explanation": "Test",
                    "parameters": {
                        "temperature": 150.0,
                        "pressure": 2.0,
                        "catalyst": "A",
                    },
                }
            ]
        )
    )

    with pytest.raises(LLMResponseError, match="Failed to recover"):
        recommender.recommend(batch_size=1, searchspace=searchspace)


@patch("baybe._optional.llm.completion")
def test_recommend_invalid_categorical(mock_completion, recommender, searchspace):
    """Invalid categorical values raise LLMResponseError after failed recovery."""
    mock_completion.return_value = _mock_response(
        json.dumps(
            [
                {
                    "explanation": "Test",
                    "parameters": {
                        "temperature": 25.0,
                        "pressure": 2.0,
                        "catalyst": "D",
                    },
                }
            ]
        )
    )

    with pytest.raises(LLMResponseError, match="Failed to recover"):
        recommender.recommend(batch_size=1, searchspace=searchspace)


@patch("baybe._optional.llm.completion")
def test_recommend_missing_parameter(mock_completion, recommender, searchspace):
    """Missing parameters raise LLMResponseError after failed recovery."""
    mock_completion.return_value = _mock_response(
        json.dumps(
            [
                {
                    "explanation": "Test",
                    "parameters": {"temperature": 25.0, "catalyst": "A"},
                }
            ]
        )
    )

    with pytest.raises(LLMResponseError, match="Failed to recover"):
        recommender.recommend(batch_size=1, searchspace=searchspace)


@patch("baybe._optional.llm.completion")
def test_recovery_success(mock_completion, recommender, searchspace):
    """Successful recovery from a malformed initial response."""
    invalid = _mock_response("Invalid JSON")
    valid = _mock_response(
        json.dumps(
            [
                {
                    "explanation": "Recovered",
                    "parameters": {
                        "temperature": 50.0,
                        "pressure": 3.0,
                        "catalyst": "C",
                    },
                }
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
        model="gpt-3.5-turbo",
        experiment_description="Test",
        objective_description="Maximize yield",
        is_feasible_experiment=lambda row: row["temperature"] > 20.0,
        overflow_experiments=1,
    )

    mock_completion.return_value = _mock_response(
        json.dumps(
            [
                {
                    "explanation": "Low temp",
                    "parameters": {
                        "temperature": 10.0,
                        "pressure": 1.0,
                        "catalyst": "A",
                    },
                },
                {
                    "explanation": "High temp",
                    "parameters": {
                        "temperature": 50.0,
                        "pressure": 2.0,
                        "catalyst": "B",
                    },
                },
                {
                    "explanation": "Med temp",
                    "parameters": {
                        "temperature": 30.0,
                        "pressure": 3.0,
                        "catalyst": "C",
                    },
                },
            ]
        )
    )

    recommendations = recommender.recommend(batch_size=2, searchspace=searchspace)

    assert len(recommendations) == 2
    assert all(recommendations["temperature"] > 20.0)


def test_initialization(recommender):
    """LLMRecommender initializes with correct attributes."""
    assert recommender.model == "gpt-3.5-turbo"
    assert recommender.experiment_description == "Test experiment"
    assert recommender.objective_description == "Maximize yield"


def test_str_representation(recommender):
    """String representation includes key information."""
    s = str(recommender)
    assert "LLMRecommender" in s
    assert "gpt-3.5-turbo" in s
