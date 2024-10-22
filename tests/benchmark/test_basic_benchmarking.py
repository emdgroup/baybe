"""Tests for the basic benchmarking classes."""

from uuid import uuid4

import pytest
from pandas import DataFrame

from benchmark.src.basic_benchmarking import (
    MultiExecutionBenchmark,
    SingleExecutionBenchmark,
)
from benchmark.src.result.basic_results import MultiResult, SingleResult


@pytest.fixture
def single_uuid():
    return uuid4()


@pytest.fixture
def multi_uuid():
    return uuid4()


@pytest.fixture
def dataframe() -> DataFrame:
    data = {
        "Num_Experiments": [1, 2, 3, 4, 5],
        "Scenario": ["A", "A", "B", "B", "C"],
        "Result": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    return DataFrame(data)


def mock_function() -> tuple[DataFrame, dict[str, str]]:
    """Future Worker needs functions on top level."""
    data = {
        "Num_Experiments": [1, 2, 3, 4, 5],
        "Scenario": ["A", "A", "B", "B", "C"],
        "Result": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    return DataFrame(data), {"meta": "data"}


@pytest.fixture
def single_benchmark(single_uuid):
    benchmark = SingleExecutionBenchmark(
        title="Single Test",
        identifier=single_uuid,
        benchmark_function=mock_function,
        metrics=[],
        objective_scenarios=[],
    )
    return benchmark


@pytest.fixture
def multi_benchmark(multi_uuid):
    benchmark = MultiExecutionBenchmark(
        title="Multi Test",
        identifier=multi_uuid,
        benchmark_function=mock_function,
        metrics=[],
        objective_scenarios=[],
        number_of_runs=4,
    )
    return benchmark


def test_single_execution_benchmark(single_benchmark, single_uuid, dataframe):
    result = single_benchmark.execute_benchmark()
    assert isinstance(result, SingleResult)
    assert result.metadata == {"meta": "data"}
    assert result.title == "Single Test"
    assert result.identifier == single_uuid
    assert result.execution_time_ns > 0
    assert result.benchmark_result.equals(dataframe)


def test_single_execution_benchmark_get_result(
    single_benchmark, single_uuid, dataframe
):
    result = single_benchmark.get_result()
    assert isinstance(result, SingleResult)
    assert result.metadata == {"meta": "data"}
    assert result.title == "Single Test"
    assert result.identifier == single_uuid
    assert result.execution_time_ns > 0
    assert result.benchmark_result.equals(dataframe)


def test_multi_execution_benchmark(multi_benchmark, multi_uuid, dataframe):
    result = multi_benchmark.execute_benchmark()
    assert isinstance(result, MultiResult)
    assert len(result.benchmark_results) == 4
    for single_result in result.benchmark_results:
        assert isinstance(single_result, SingleResult)
        assert single_result.metadata == {"meta": "data"}
        assert single_result.title == "Multi Test"
        assert single_result.identifier == multi_uuid
        assert single_result.execution_time_ns > 0
        assert single_result.benchmark_result.equals(dataframe)


def test_multi_execution_benchmark_get_result(multi_benchmark, multi_uuid, dataframe):
    result = multi_benchmark.get_result()
    assert isinstance(result, MultiResult)
    assert len(result.benchmark_results) == 4
    for single_result in result.benchmark_results:
        assert isinstance(single_result, SingleResult)
        assert single_result.metadata == {"meta": "data"}
        assert single_result.title == "Multi Test"
        assert single_result.identifier == multi_uuid
        assert single_result.execution_time_ns > 0
        assert single_result.benchmark_result.equals(dataframe)
