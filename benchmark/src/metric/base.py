"""Base classes for benchmarking metrics."""

from abc import ABC, abstractmethod

from attrs import define
from pandas import DataFrame


@define
class Metric(ABC):
    """Abstract base class for all benchmarking metrics."""

    @abstractmethod
    def evaluate(
        self, prediction: DataFrame, objective_scenarios: list[str] | None = None
    ) -> dict[str, float]:
        """Evaluate the given predictions against the objective scenario.

        Args:
            prediction: The predictions to evaluate.
                        objective_scenarios: The scenario names to calculate the
                        metric and apply the thresholds if set.
            objective_scenarios: The scenario names to calculate the metric.
                                 must match the the defined names in the
                                 :func:`baybe.simulation.scenarios.simulate_scenarios`
                                 scenarios dict.

        Returns:
            dict[str, float]: A dictionary containing evaluation metrics as keys
            and their corresponding values.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of the metric.

        Returns:
            str: A readable metric name.
        """
        pass

    @abstractmethod
    def _check_threshold(self, values: dict[str, float]) -> None:
        """Check if the threshold is met.

        Args:
            values: The dictionary containing the scenario names and their values.
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
