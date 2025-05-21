"""Tests for the LLM-based recommender."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from baybe.exceptions import LLMResponseError
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
)
from baybe.recommenders.pure.llm.llm import (
    LLMRecommender,
    ParameterDescription,
    ParameterType,
)
from baybe.searchspace import SearchSpace


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
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
            )
        ]
    )


@pytest.fixture
def parameter_descriptions():
    """Create parameter descriptions for testing."""
    return [
        ParameterDescription(
            name="temperature",
            description="Reaction temperature",
            type=ParameterType.CONTINUOUS,
            bounds=(0.0, 100.0),
            unit="°C",
        ),
        ParameterDescription(
            name="pressure",
            description="Reaction pressure",
            type=ParameterType.CONTINUOUS,
            bounds=(0.0, 5.0),
            unit="bar",
        ),
        ParameterDescription(
            name="catalyst",
            description="Catalyst type",
            type=ParameterType.DISCRETE_CATEGORICAL,
            bounds=["A", "B", "C"],
        ),
    ]


@pytest.fixture
def searchspace():
    """Create a search space for testing."""
    parameters = [
        NumericalContinuousParameter(
            name="temperature",
            bounds=(0.0, 100.0),
        ),
        NumericalContinuousParameter(
            name="pressure",
            bounds=(0.0, 5.0),
        ),
        CategoricalParameter(
            name="catalyst",
            values=["A", "B", "C"],
        ),
    ]
    return SearchSpace.from_product(parameters)


@pytest.fixture
def llm_recommender(parameter_descriptions):
    """Create an LLMRecommender instance for testing."""
    return LLMRecommender(
        model="gpt-3.5-turbo",
        experiment_description="Test experiment",
        objective_description="Maximize yield",
        parameter_descriptions=parameter_descriptions,
    )


def test_llm_recommender_initialization(llm_recommender, parameter_descriptions):
    """Test that LLMRecommender initializes correctly."""
    assert llm_recommender.model == "gpt-3.5-turbo"
    assert llm_recommender.experiment_description == "Test experiment"
    assert llm_recommender.objective_description == "Maximize yield"
    assert llm_recommender.parameter_descriptions == parameter_descriptions


@patch("baybe.recommenders.pure.llm.llm.completion")
def test_recommend_success(
    mock_completion, llm_recommender, searchspace, mock_llm_response
):
    """Test successful recommendation generation."""
    mock_completion.return_value = mock_llm_response

    recommendations = llm_recommender.recommend(
        batch_size=2,
        searchspace=searchspace,
    )

    assert isinstance(recommendations, pd.DataFrame)
    assert len(recommendations) == 2
    assert list(recommendations.columns) == ["temperature", "pressure", "catalyst"]
    assert recommendations["temperature"].tolist() == [25.0, 30.0]
    assert recommendations["pressure"].tolist() == [2.0, 1.5]
    assert recommendations["catalyst"].tolist() == ["A", "B"]


@patch("baybe.recommenders.pure.llm.llm.completion")
def test_recommend_with_measurements(
    mock_completion, llm_recommender, searchspace, mock_llm_response
):
    """Test recommendation generation with previous measurements."""
    mock_completion.return_value = mock_llm_response

    measurements = pd.DataFrame(
        {
            "temperature": [20.0, 25.0],
            "pressure": [1.0, 2.0],
            "catalyst": ["A", "B"],
            "yield": [0.5, 0.7],
        }
    )

    recommendations = llm_recommender.recommend(
        batch_size=2,
        searchspace=searchspace,
        measurements=measurements,
    )

    assert isinstance(recommendations, pd.DataFrame)
    assert len(recommendations) == 2


@patch("baybe.recommenders.pure.llm.llm.completion")
def test_recommend_with_invalid_response(mock_completion, llm_recommender, searchspace):
    """Test handling of invalid LLM response."""
    # Mock an invalid JSON response
    mock_completion.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Invalid JSON"))]
    )

    with pytest.raises(LLMResponseError):
        llm_recommender.recommend(
            batch_size=2,
            searchspace=searchspace,
        )


@patch("baybe.recommenders.pure.llm.llm.completion")
def test_recommend_with_out_of_bounds_values(
    mock_completion, llm_recommender, searchspace
):
    """Test handling of out-of-bounds parameter values."""
    # Mock response with out-of-bounds temperature
    mock_completion.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        [
                            {
                                "explanation": "Test suggestion",
                                "parameters": {
                                    "temperature": 150.0,  # Out of bounds
                                    "pressure": 2.0,
                                    "catalyst": "A",
                                },
                            }
                        ]
                    )
                )
            )
        ]
    )

    with pytest.raises(LLMResponseError):
        llm_recommender.recommend(
            batch_size=1,
            searchspace=searchspace,
        )


@patch("baybe.recommenders.pure.llm.llm.completion")
def test_recommend_with_invalid_categorical_value(
    mock_completion, llm_recommender, searchspace
):
    """Test handling of invalid categorical parameter values."""
    # Mock response with invalid catalyst value
    mock_completion.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        [
                            {
                                "explanation": "Test suggestion",
                                "parameters": {
                                    "temperature": 25.0,
                                    "pressure": 2.0,
                                    "catalyst": "D",  # Invalid value
                                },
                            }
                        ]
                    )
                )
            )
        ]
    )

    with pytest.raises(LLMResponseError):
        llm_recommender.recommend(
            batch_size=1,
            searchspace=searchspace,
        )


@patch("baybe.recommenders.pure.llm.llm.completion")
def test_recommend_with_missing_parameter(
    mock_completion, llm_recommender, searchspace
):
    """Test handling of missing parameters in response."""
    # Mock response missing the pressure parameter
    mock_completion.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        [
                            {
                                "explanation": "Test suggestion",
                                "parameters": {
                                    "temperature": 25.0,
                                    "catalyst": "A",
                                },
                            }
                        ]
                    )
                )
            )
        ]
    )

    with pytest.raises(LLMResponseError):
        llm_recommender.recommend(
            batch_size=1,
            searchspace=searchspace,
        )
