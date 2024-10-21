"""This module contains the benchmarks for the different domains of the benchmark."""

from benchmark.domain.direct_arylation import benchmark_direct_arylation
from benchmark.domain.synthetic_1 import benchmark_synthetic_1
from benchmark.src import SingleExecutionBenchmark

SINGE_BENCHMARKS_TO_RUN: list[SingleExecutionBenchmark] = [
    benchmark_synthetic_1,
    benchmark_direct_arylation,
]

__all__ = ["SINGE_BENCHMARKS_TO_RUN"]
