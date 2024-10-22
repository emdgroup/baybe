"""Tests for the basic_results module."""

from uuid import UUID, uuid4

import pandas as pd
import pytest
from matplotlib.figure import Figure
from pandas import DataFrame

from benchmark.src.metric import Metric
from benchmark.src.result.basic_results import SingleResult


@pytest.fixture
def sample_dataframe():
    data = {
        "Num_Experiments": [1, 2, 3, 4, 5],
        "Scenario": ["A", "A", "B", "B", "C"],
        "Result": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    return pd.DataFrame(data)


@pytest.fixture
def uuid() -> UUID:
    return uuid4()


@pytest.fixture
def title() -> str:
    return "title"


@pytest.fixture
def metadata() -> dict[str, str]:
    return {"meta": "data"}


class MockMetric(Metric):
    """A mock metric for testing."""

    def evaluate(
        self, result: DataFrame, objective_scenario: list[str]
    ) -> dict[str, float]:
        return {"dummy_metric": 1.0}

    def __str__(self) -> str:
        return "Dummy Metric"

    def __repr__(self) -> str:
        return "Dummy Metric"

    def _check_threshold(self, result: dict[str, float]) -> None:
        pass


@pytest.fixture
def single_result(title, sample_dataframe, uuid, metadata):
    return SingleResult(
        title=title,
        identifier=uuid,
        metadata=metadata,
        benchmark_result=sample_dataframe,
        execution_time_ns=1000,
    )


def test_get_execution_time_ns(single_result):
    assert single_result.get_execution_time_ns() == 1000


def test_create_convergence_plot(single_result):
    plot = single_result.create_convergence_plot()
    assert isinstance(plot, Figure)


def test_evaluate_result(single_result):
    metric = MockMetric()
    result = single_result.evaluate_result(metric, ["A", "B", "C"])
    assert result == {"dummy_metric": 1.0}


def test_to_csv(single_result):
    csv_string = single_result.to_csv()
    assert "Num_Experiments,Scenario,Result" in csv_string
    assert "1,A,0.1" in csv_string

    single_result.to_csv("test_output.csv")
    with open("test_output.csv") as file:
        content = file.read()
        assert "Num_Experiments,Scenario,Result" in content
        assert "1,A,0.1" in content
