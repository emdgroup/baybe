"""This module contains the configuration of a benchmark scenario."""

from attrs import define, field
from attrs.validators import deep_mapping, instance_of

from baybe.recommenders.base import RecommenderProtocol
from baybe.serialization.mixin import SerialMixin


@define(frozen=True)
class BenchmarkScenarioSettings(SerialMixin):
    """The configuration for the benchmark settings."""

    batch_size: int = field(validator=instance_of(int))
    """The batch size for the benchmark."""

    n_doe_iterations: int = field(validator=instance_of(int))
    """The number of Design of Experiment iterations."""

    n_mc_iterations: int = field(validator=instance_of(int))
    """The number of Monte Carlo iterations."""

    recommender: dict[str, RecommenderProtocol] = field(
        validator=deep_mapping(
            instance_of(str), instance_of(RecommenderProtocol), instance_of(dict)
        )
    )
    """The recommender to use for the benchmark.
    The key is the name which will identify the recommenders campaign."""
