"""Benchmarking module for executing and comparing performance related tasks."""

from benchmark.base import Benchmark
from benchmark.basic import MultiExecutionBenchmark, SingleExecutionBenchmark
from benchmark.metric import (
    Metric,
    NormalizedNegativeRootMeanSquaredErrorMetric,
)
from benchmark.result import Result
from benchmark.result.basic import MultiResult, SingleResult

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
]
