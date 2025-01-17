"""Benchmark domains."""

from benchmarks.definition.config import Benchmark
from benchmarks.domains.synthetic_2C1D_1C import synthetic_2C1D_1C_benchmark
from benchmarks.domains.synthetic_michalewicz import synthetic_michalewicz

BENCHMARKS: list[Benchmark] = [
    synthetic_2C1D_1C_benchmark,
    synthetic_michalewicz,
]

__all__ = ["BENCHMARKS"]
