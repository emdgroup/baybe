"""This module contains the benchmarks for the different domains of the benchmark."""

from benchmark.domain.direct_arylation import benchmark_direct_arylation
from benchmark.domain.synthetic_1 import benchmark_synthetic_1
from benchmark.domain.synthetic_2 import benchmark_synthetic_2
from benchmark.domain.synthetic_3 import benchmark_synthetic_3
from benchmark.domain.transfer_learning_backtesting import (
    benchmark_transfer_learning_backtesting,
)
from benchmark.src import SingleExecutionBenchmark

SINGE_BENCHMARKS_TO_RUN: list[SingleExecutionBenchmark] = [
    benchmark_synthetic_1,
    benchmark_direct_arylation,
    benchmark_synthetic_2,
    benchmark_synthetic_3,
    benchmark_transfer_learning_backtesting,
]

__all__ = ["SINGE_BENCHMARKS_TO_RUN"]
