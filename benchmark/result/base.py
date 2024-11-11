"""Base classes for the benchmarking results."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from attrs import define, field

from benchmark.metric.base import Metric


@define(frozen=True)
class Result(ABC):
    """Abstract base class for all benchmarking results."""

    @staticmethod
    def _convert_metadata_to_string(metadata: dict[Any, Any]) -> dict[str, str]:
        """Convert the metadata to a string representation.

        The function will convert the metadata to a string representation
        to ensure that the metadata can be written to a csv file.
        """
        metadata_return: dict[str, str] = dict()
        for key, value in metadata.items():
            sanitized_key = str(key).replace(" ", "_")
            metadata_return[sanitized_key] = str(value)
        return metadata_return

    title: str = field()
    """The title of the benchmarking result."""

    identifier: UUID = field()
    """The unique identifier of the benchmark running which can be set
    to compare different executions of the same benchmark setting."""

    metadata: dict[str, str] = field(converter=_convert_metadata_to_string)
    """Metadata about the benchmarking result."""

    @abstractmethod
    def get_execution_time_ns(self) -> float:
        """Return the execution time of the benchmark in nanoseconds."""
        pass

    @abstractmethod
    def evaluate_result(
        self, metric: Metric, objective_scenario: list[str]
    ) -> dict[str, float]:
        """Evaluate the result based on the given metric and objective scenario.

        Args:
            metric: The metric to be used for evaluation.
            objective_scenario: The scenario name to be used for evaluation.

        Returns:
            float: The evaluation result as a floating-point number.
        """
        pass

    @abstractmethod
    def to_csv(self, path: str | None = None) -> str | None:
        """Write the result to a csv file.

        Writes the result to a csv file. If no path is given, the function
        will return the csv as a string.
        """
        pass
