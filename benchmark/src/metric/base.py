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
        """Return the string representation of the metric.

        Returns:
            str: A readable metric name.
        """
        pass


@define
class GoalOrientedMetricInterface(ABC):
    """Interface for goal-oriented metrics."""

    @abstractmethod
    def get_objective_value(self) -> float:
        """Retrieve the objective value.

        This method return the objective value of the
        metric based in the metric (e.g., Min or Max).

        Returns:
            float: The objective value.
        """
        pass


@define
class NormalizationInterface(ABC):
    """Interface for normalization."""

    @abstractmethod
    def _normalize_data(self, data: DataFrame, index_name: str) -> DataFrame:
        """Normalize the given data to scale the metric to [0,1].

        Args:
            data: The data to be normalized.
            index_name: The name of the index to be used for normalization.

        Returns:
            DataFrame: The normalized data.
        """
        pass
