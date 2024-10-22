"""Benchmarking module for executing and comparing performance related tasks."""

from benchmark.src.base import Benchmark
from benchmark.src.basic_benchmarking import (
    MultiExecutionBenchmark,
    SingleExecutionBenchmark,
)
from benchmark.src.metric import (
    Metric,
    NormalizedAreaUnderTheCurve,
)
from benchmark.src.persistance import (
    LocalExperimentResultPersistence,
    ResultPersistenceInterface,
    S3ExperimentResultPersistence,
)
from benchmark.src.result import Result
from benchmark.src.result.basic_results import MultiResult, SingleResult

__all__ = [
    "Benchmark",
    "MultiExecutionBenchmark",
    "SingleExecutionBenchmark",
    "Result",
    "MultiResult",
    "SingleResult",
    "Metric",
    "NormalizedAreaUnderTheCurve",
    "ResultPersistenceInterface",
    "LocalExperimentResultPersistence",
    "S3ExperimentResultPersistence",
]
