"""Benchmark domains."""

from benchmarks.definition.base import Benchmark
from benchmarks.domains.synthetic_2C1D_1C import synthetic_2C1D_1C_benchmark
from benchmarks.domains.CrabNet_AdvOpt import crabnet_benchmark, crabnet_transfer_learning_benchmark

BENCHMARKS: list[Benchmark] = [
    synthetic_2C1D_1C_benchmark,
    crabnet_benchmark,
    crabnet_transfer_learning_benchmark,
]

__all__ = ["BENCHMARKS"]
