"""Benchmark configurations."""

from collections.abc import Callable
from typing import TypeVar

from attrs import define, field
from attrs.validators import instance_of, is_callable
from pandas import DataFrame

from baybe.serialization.mixin import SerialMixin

BenchmarkSettings = TypeVar("BenchmarkSettings")

BenchmarkFunction = Callable[[BenchmarkSettings, Callable | DataFrame], DataFrame]


@define(frozen=True)
class BenchmarkFunctionDefinition:
    """The core components including the benchmark function."""

    callable: BenchmarkFunction = field(validator=is_callable())
    """The callable that executes the benchmark code and returns the result."""

    description: str = field(validator=instance_of(str))
    """The description of the benchmark function."""

    best_possible_result: float | None = field(default=None)
    """The best possible result which can be achieved in the optimization process."""


@define(frozen=True)
class ConvergenceExperimentSettings(SerialMixin):
    """Benchmark configuration for recommender convergence analyses."""

    batch_size: int = field(validator=instance_of(int))
    """The recommendation batch size."""

    n_doe_iterations: int = field(validator=instance_of(int))
    """The number of Design of Experiment iterations."""

    n_mc_iterations: int = field(validator=instance_of(int))
    """The number of Monte Carlo iterations."""

    random_seed: int = field(validator=instance_of(int), default=1337)
    """The random seed for reproducibility."""
