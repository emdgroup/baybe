"""Benchmark domains."""

from benchmarks.definition.base import Benchmark
from benchmarks.domains.synthetic_2C1D_1C import synthetic_2C1D_1C_benchmark
from benchmarks.domains.synthetic_michalewicz import synthetic_michalewicz_benchmark

BENCHMARKS: list[Benchmark] = [
    synthetic_2C1D_1C_benchmark,
    synthetic_michalewicz_benchmark,
]

__all__ = ["BENCHMARKS"]
