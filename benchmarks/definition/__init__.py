"""Benchmark definitions."""

from benchmarks.definition.base import Benchmark, BenchmarkSettings, RunMode
from benchmarks.definition.convergence import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.definition.regression import (
    RegressionBenchmark,
    RegressionBenchmarkSettings,
    TransferLearningRegressionBenchmark,
    TransferLearningRegressionBenchmarkSettings,
)

__all__ = [
    "Benchmark",
    "BenchmarkSettings",
    "ConvergenceBenchmark",
    "ConvergenceBenchmarkSettings",
    "RegressionBenchmark",
    "RegressionBenchmarkSettings",
    "TransferLearningRegressionBenchmark",
    "TransferLearningRegressionBenchmarkSettings",
    "RunMode",
]
