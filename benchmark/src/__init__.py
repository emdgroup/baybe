"""Benchmarking module for executing and comparing performance related tasks."""

from src.base import Benchmark
from src.basic import MultiExecutionBenchmark, SingleExecutionBenchmark
from src.metric import (
    Metric,
    NormalizedNegativeRootMeanSquaredErrorMetric,
)
from src.persistance import (
    LocalExperimentResultPersistence,
    ResultPersistenceInterface,
    S3ExperimentResultPersistence,
)
from src.result import Result
from src.result.basic import MultiResult, SingleResult

__all__ = [
    "Benchmark",
    "MultiExecutionBenchmark",
    "SingleExecutionBenchmark",
    "Result",
    "MultiResult",
    "SingleResult",
    "Metric",
    "NormalizedGoalOrientationMetric",
    "NormalizedNegativeRootMeanSquaredErrorMetric",
    "ResultPersistenceInterface",
    "LocalExperimentResultPersistence",
    "S3ExperimentResultPersistence",
]
