"""Benchmarking module for executing and comparing performance related tasks."""

from benchmark.definition.base import Benchmark
from benchmark.definition.basic_benchmarking import SingleExecutionBenchmark
from benchmark.metric.base import Metric
from benchmark.metric.simple_regret import NormalizedSimpleRegret
from benchmark.result.base import Result
from benchmark.result.basic_results import SingleResult

__all__ = [
    "Benchmark",
    "SingleExecutionBenchmark",
    "Result",
    "SingleResult",
    "Metric",
    "NormalizedSimpleRegret",
]
