"""Benchmark configurations."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from attrs import define, field
from attrs.validators import instance_of
from pandas import DataFrame

from baybe.serialization.mixin import SerialMixin


@define(frozen=True)
class BenchmarkSettings(SerialMixin, ABC):
    """Benchmark configuration for recommender analyses."""

    random_seed: int = field(validator=instance_of(int), kw_only=True, default=1337)
    """The random seed for reproducibility."""


@define(frozen=True)
class ConvergenceExperimentSettings(BenchmarkSettings):
    """Benchmark configuration for recommender convergence analyses."""

    batch_size: int = field(validator=instance_of(int))
    """The recommendation batch size."""

    n_doe_iterations: int = field(validator=instance_of(int))
    """The number of Design of Experiment iterations."""

    n_mc_iterations: int = field(validator=instance_of(int))
    """The number of Monte Carlo iterations."""


FunctionSettings = TypeVar("FunctionSettings", bound=BenchmarkSettings)


@define(frozen=True)
class BenchmarkExecutableBase(ABC, Generic[FunctionSettings]):
    """The base class for a benchmark executable."""

    settings: FunctionSettings = field()
    """The benchmark configuration."""

    lookup: Callable | DataFrame = field()
    """The lookup function or DataFrame for the benchmark."""

    best_possible_result: float | None = field(default=None)
    """The best possible result which can be achieved in the optimization process."""

    optimal_function_inputs: list[dict[str, Any]] | None = field(default=None)

    @property
    def description(self) -> str:
        """The description of the benchmark function."""
        if callable(self.lookup) and self.lookup.__doc__ is not None:
            return self.lookup.__doc__
        if self.__call__.__doc__ is not None:
            return self.__call__.__doc__
        return (
            f"Best possible result: {self.best_possible_result} with"
            + "the input(s): {self.optimal_function_inputs}"
        )

    @settings.validator
    def _validate_settings(self, _, value: FunctionSettings) -> None:
        """Validate the settings."""
        if not isinstance(value, BenchmarkSettings):
            raise ValueError(
                f"Invalid settings '{value}'. Must be a BenchmarkSettings."
            )

    @lookup.validator
    def _validate_lookup(self, _, value: Callable | DataFrame) -> None:
        """Validate the lookup."""
        if not callable(value) and not isinstance(value, DataFrame):
            raise ValueError(
                f"Invalid lookup '{value}'. Must be a Callable or DataFrame."
            )

    @abstractmethod
    def __call__(self) -> DataFrame:
        """Execute the benchmark and return the result."""
        raise NotImplementedError
