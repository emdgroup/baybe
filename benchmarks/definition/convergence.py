"""Convergence benchmark configuration."""

from typing import Any

from attrs import define, field, fields
from attrs.validators import deep_mapping, instance_of, optional

from benchmarks.definition.base import (
    Benchmark,
    BenchmarkSettings,
    RunMode,
    make_runmode_attr_validator,
)


@define(kw_only=True, frozen=True)
class ConvergenceBenchmarkSettings(BenchmarkSettings):
    """Benchmark configuration for recommender convergence analyses."""

    batch_size_settings: dict[RunMode, int] = field(
        validator=make_runmode_attr_validator(int)
    )
    """The settings for recommendation batch size."""

    n_doe_iterations_settings: dict[RunMode, int] = field(
        validator=make_runmode_attr_validator(int)
    )
    """The settings for number of Design of Experiment iterations."""

    n_mc_iterations_settings: dict[RunMode, int] = field(
        validator=make_runmode_attr_validator(int)
    )
    """The settings for number of Monte Carlo iterations."""

    @property
    def batch_size(self) -> int:
        """The batch size for the current runmode."""
        if self.runmode not in self.batch_size_settings:
            raise ValueError(
                "Current runmode not found in "
                f"{fields(ConvergenceBenchmarkSettings).batch_size_settings.name}"
                f": {self.runmode.value}"
            )
        return self.batch_size_settings[self.runmode]

    @property
    def n_doe_iterations(self) -> int:
        """The number of Design of Experiment iterations for the current runmode."""
        if self.runmode not in self.n_doe_iterations_settings:
            raise ValueError(
                "Current runmode not found in "
                f"{fields(ConvergenceBenchmarkSettings).n_doe_iterations_settings.name}"
                f": {self.runmode.value}"
            )
        return self.n_doe_iterations_settings[self.runmode]

    @property
    def n_mc_iterations(self) -> int:
        """The number of Monte Carlo iterations for the current runmode."""
        if self.runmode not in self.n_mc_iterations_settings:
            raise ValueError(
                "Current runmode not found in "
                f"{fields(ConvergenceBenchmarkSettings).n_mc_iterations_settings.name}"
                f": {self.runmode.value}"
            )
        return self.n_mc_iterations_settings[self.runmode]


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
