"""Benchmark definitions."""

from benchmarks.definition.base import (
    Benchmark,
    BenchmarkSettings,
)
from benchmarks.definition.convergence import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.definition.regression import (
    RegressionBenchmark,
    RegressionBenchmarkSettings,
)

__all__ = [
    "Benchmark",
    "BenchmarkSettings",
    "ConvergenceBenchmark",
    "ConvergenceBenchmarkSettings",
    "RegressionBenchmark",
    "RegressionBenchmarkSettings",
]
