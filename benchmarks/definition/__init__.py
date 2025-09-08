"""Benchmark definitions."""

from benchmarks.definition.base import Benchmark, BenchmarkSettings
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
from benchmarks.definition.utils import RunMode

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
