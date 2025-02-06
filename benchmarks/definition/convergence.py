"""Convergence benchmark configuration."""

from typing import Any

from attrs import define, field
from attrs.validators import deep_mapping, instance_of, optional

from benchmarks.definition.base import Benchmark, BenchmarkSettings


@define(frozen=True, kw_only=True)
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

    optimal_target_values: dict[str, Any] | None = field(
        default=None,
        validator=optional(
            deep_mapping(
                key_validator=instance_of(str),
                mapping_validator=instance_of(dict),
                value_validator=lambda *_: None,
            )
        ),
    )
    """The optimal values that can be achieved for the targets **individually**."""
