"""Benchmarking module for performance tracking."""

from benchmarks.definition import (
    Benchmark,
    BenchmarkSettings,
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.result import Result
from benchmarks.utils import (
    create_compare_plot,
    create_compare_plot_from_paths,
    load_benchmark_results,
)

__all__ = [
    "Benchmark",
    "BenchmarkSettings",
    "ConvergenceBenchmark",
    "ConvergenceBenchmarkSettings",
    "Result",
    "create_compare_plot",
    "load_benchmark_results",
    "create_compare_plot_from_paths",
]
