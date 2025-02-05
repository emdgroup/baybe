"""Convergence benchmark configuration."""

from typing import Any

from attrs import define, field
from attrs.validators import instance_of

from benchmarks.definition.config import Benchmark, BenchmarkSettings


@define(frozen=True)
class ConvergenceBenchmarkSettings(BenchmarkSettings):
    """Benchmark configuration for recommender convergence analyses."""

    batch_size: int = field(validator=instance_of(int))
    """The recommendation batch size."""

    n_doe_iterations: int = field(validator=instance_of(int))
    """The number of Design of Experiment iterations."""

    n_mc_iterations: int = field(validator=instance_of(int))
    """The number of Monte Carlo iterations."""


@define(frozen=True)
class ConvergenceBenchmark(Benchmark[ConvergenceBenchmarkSettings]):
    """A class for defining convergence benchmarks."""

    best_possible_result: float | None = field(default=None)
    """The best possible result which can be achieved in the optimization process."""

    optimal_function_inputs: list[dict[str, Any]] | None = field(default=None)
    """An input that creates the best_possible_result."""
