"""Base classes for benchmarking metrics."""

from abc import ABC, abstractmethod

from attrs import define
from pandas import DataFrame


@define
class Metric(ABC):
    """Abstract base class for all benchmarking metrics."""

    @abstractmethod
    def evaluate(self, prediction: DataFrame) -> float:
        """Evaluate the benchmarking metric and return the result."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of the metric."""
        pass


@define
class GoalOrientedMetricInterface(ABC):
    """Interface for goal-oriented metrics."""

    @abstractmethod
    def get_objective_value(self) -> float:
        """Get the objective value from the lookup table."""
        pass


@define
class NormalizationInterface(ABC):
    """Interface for normalization."""

    @abstractmethod
    def _normalize_data(self, data: DataFrame, index_name: str) -> DataFrame:
        """Normalize the data."""
        pass
