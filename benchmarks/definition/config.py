"""Benchmark configurations."""

from typing import TypeVar

from attrs import define, field
from attrs.validators import instance_of

from baybe.serialization.mixin import SerialMixin

BenchmarkSettings = TypeVar("BenchmarkSettings")


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
