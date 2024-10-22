"""Basic result classes for benchmarking."""

from attrs import define, field
from pandas import DataFrame
from typing_extensions import override

from benchmark.metric.base import Metric
from benchmark.result.base import Result


@define(frozen=True)
class SingleResult(Result):
    """A single result of the benchmarking."""

    benchmark_result: DataFrame = field()
    """The result of the benchmarked callable."""

    execution_time_ns: int = field()
    """The execution time of the benchmark in nanoseconds."""

    @override
    def get_execution_time_ns(self) -> float:
        """Return the execution time of the benchmark in nanoseconds."""
        return self.execution_time_ns

    @override
    def evaluate_result(
        self, metric: Metric, objective_scenario: list[str]
    ) -> dict[str, float]:
        """Evaluate the benchmarking result using the given metric.

        See :func:`benchmark.result.base.Result.evaluate_result` for more information.
        """
        metric_value = metric.evaluate(self.benchmark_result, objective_scenario)
        return metric_value

    @override
    def to_csv(self, path: str | None = None) -> str | None:
        """Write the result to a csv file.

        Writes the result to a csv file. If no path is given, the function
        will return the csv as a string.

        Args:
            path: Optional path to write the csv file to. If None, the function
                will return the csv as a string.

        Returns:
            str | None: The csv as a string if no path is given.
        """
        if path is None:
            return self.benchmark_result.to_csv(index=False)
        self.benchmark_result.to_csv(path, index=False)
        return None
