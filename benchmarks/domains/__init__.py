<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
"""Benchmark domains."""

from benchmarks.definition.base import Benchmark
from benchmarks.domains.synthetic_2C1D_1C import synthetic_2C1D_1C_benchmark
from benchmarks.domains.Hardness import hardness_benchmark, hardness_transfer_learning_benchmark

BENCHMARKS: list[Benchmark] = [
    synthetic_2C1D_1C_benchmark,
    hardness_benchmark,
    hardness_transfer_learning_benchmark, 
]

__all__ = ["BENCHMARKS"]
=======
"""Benchmark domains."""

from benchmarks.definition.config import Benchmark
from benchmarks.domains.Hardness import (
    hardness_benchmark,
    hardness_transfer_learning_benchmark,
)
from benchmarks.domains.synthetic_2C1D_1C import synthetic_2C1D_1C_benchmark

BENCHMARKS: list[Benchmark] = [
    synthetic_2C1D_1C_benchmark,
    hardness_benchmark,
    hardness_transfer_learning_benchmark,
]

__all__ = ["BENCHMARKS"]
>>>>>>> 14473d7b ([pre-commit.ci] auto fixes from pre-commit.com hooks)
=======
"""Benchmark domains."""

from benchmarks.definition.base import Benchmark
from benchmarks.domains.Hardness import (
    hardness_benchmark,
    hardness_transfer_learning_benchmark,
)
from benchmarks.domains.synthetic_2C1D_1C import synthetic_2C1D_1C_benchmark

BENCHMARKS: list[Benchmark] = [
    synthetic_2C1D_1C_benchmark,
    hardness_benchmark,
    hardness_transfer_learning_benchmark,
]

__all__ = ["BENCHMARKS"]
>>>>>>> 0a265eec (Rename config module to base)
=======
"""Benchmark domains."""

from benchmarks.definition.base import Benchmark
from benchmarks.domains.synthetic_2C1D_1C import synthetic_2C1D_1C_benchmark
from benchmarks.domains.Hardness import hardness_benchmark, hardness_transfer_learning_benchmark

BENCHMARKS: list[Benchmark] = [
    synthetic_2C1D_1C_benchmark,
    hardness_benchmark,
    hardness_transfer_learning_benchmark,
]

__all__ = ["BENCHMARKS"]
>>>>>>> 0a8aed51 (Update __init__.py)
=======
"""Benchmark domains."""

from benchmarks.definition.base import Benchmark
from benchmarks.domains.Hardness import (
    hardness_benchmark,
    hardness_transfer_learning_benchmark,
)
from benchmarks.domains.synthetic_2C1D_1C import synthetic_2C1D_1C_benchmark

BENCHMARKS: list[Benchmark] = [
    synthetic_2C1D_1C_benchmark,
    hardness_benchmark,
    hardness_transfer_learning_benchmark,
]

__all__ = ["BENCHMARKS"]
>>>>>>> 59c30efb ([pre-commit.ci] auto fixes from pre-commit.com hooks)
