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
            prediction: The predictions to evaluate from the an benchmark result.

            objective_scenarios: The scenario names to calculate the metric.
                Must match the defined names in the
                :func:`baybe.simulation.scenarios.simulate_scenarios`
                scenarios dict. Only the appearing names will be evaluated.
                If None all scenarios will be evaluated.

        Returns:
            A dictionary containing evaluation metrics as keys
            and their corresponding values.
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
