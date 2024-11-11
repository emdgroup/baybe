"""Benchmark configurations."""

from typing import TypeVar

from attrs import define, field
from attrs.validators import deep_mapping, instance_of

from baybe.recommenders.base import RecommenderProtocol
from baybe.serialization.mixin import SerialMixin

BenchmarkConfig = TypeVar("BenchmarkConfig", bound="SerialMixin")


@define(frozen=True)
class RecommenderConvergenceAnalysis(SerialMixin):
    """Benchmark configuration for recommender convergence analyses."""

    recommenders: dict[str, RecommenderProtocol] = field(
        validator=deep_mapping(
            key_validator=instance_of(str),
            value_validator=instance_of(RecommenderProtocol),
            mapping_validator=instance_of(dict),
        )
    )
    """The recommenders to compare (keys act as labels in the benchmark result)."""

    batch_size: int = field(validator=instance_of(int))
    """The recommendation batch size."""

    n_doe_iterations: int = field(validator=instance_of(int))
    """The number of Design of Experiment iterations."""

    n_mc_iterations: int = field(validator=instance_of(int))
    """The number of Monte Carlo iterations."""
