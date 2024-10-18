"""Base classes for the benchmarking results."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from attrs import define, field
from matplotlib.figure import Figure, SubFigure

from benchmark.metric import Metric


@define(frozen=True)
class Result(ABC):
    """Abstract base class for all benchmarking results."""

    @staticmethod
    def _convert_metadata_to_string(metadata: dict[Any, Any]) -> dict[str, str]:
        """Convert the metadata to a string representation.

        The function will convert the metadata to a string representation
        to ensure that the metadata can be written to a csv file.
        """
        return {str(key): str(value) for key, value in metadata.items()}

    title: str
    """The title of the benchmarking result."""

    identifier: UUID
    """The unique identifier of the benchmark running which can be set
    to compare different executions of the same benchmark setting."""

    metadata: dict[str, str] = field(converter=_convert_metadata_to_string)
    """Metadata about the benchmarking result."""

    @abstractmethod
    def get_execution_time_ns(self) -> float:
        """Return the execution time of the benchmark in nanoseconds."""
        pass

    @abstractmethod
    def create_convergence_plot(self) -> Figure | SubFigure:
        """Generate a plot of the benchmarking result.

        The function will create a mathplotlib figure or subfigure
        from the benchmarking result. The function assumes that the
        result contains a column with the name `Num_Experiments`, a
        column with the name `Scenario`, and that the last column
        contains the best found value over the experiments.
        """
        pass

    @abstractmethod
    def evaluate_result(self, metric: Metric) -> float:
        """Evaluate the benchmarking result using the given metric.

        The function will evaluate the benchmarking result using the
        given metric. The metric may throw an exception if a threshold
        is set and the result does not meet the threshold.
        """
        pass

    @abstractmethod
    def to_csv(self, path: str = None) -> str | None:
        """Write the result to a csv file.

        Writes the result to a csv file. If no path is given, the function
        will return the csv as a string.
        """
        pass
