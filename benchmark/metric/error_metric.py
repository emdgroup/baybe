"""Basic error metric implementations for benchmarking."""

from attrs import define, field
from pandas import DataFrame
from typing_extensions import override

from baybe.targets import TargetMode
from benchmark.metric.base import (
    GoalOrientedMetricInterface,
    Metric,
    NormalizationInterface,
)


@define
class NormalizedNegativeRootMeanSquaredErrorMetric(
    Metric, GoalOrientedMetricInterface, NormalizationInterface
):
    """Normalized negative root mean squared error metric for regression tasks."""

    lookup: DataFrame
    """The lookup table or function to evaluate the goal orientation
    metric and compare the best included result."""

    objective_name: str
    """The name of the objective to evaluate."""

    target_mode_to_eval: TargetMode = field(converter=TargetMode)
    """The target mode of the optimization under evaluation."""

    threshold: float = field(default=None)
    """Optional threshold for the metric.
    If the metric is below the threshold, an exception is thrown."""

    @override
    def _normalize_data(self, data: DataFrame, index_name: str) -> DataFrame:
        """Normalize the column with the best found value so far for comparison."""
        max_value = data[index_name].max()
        min_value = data[index_name].min()
        data[index_name] = data[index_name].apply(
            lambda x: (x - min_value) / (max_value - min_value)
        )
        return data

    def __attrs_post_init__(self):
        if self.target_mode_to_eval == TargetMode.MATCH:
            raise ValueError("Matching target mode not yet supported.")
        normalized_lookup = self.lookup.copy()
        normalized_lookup = self._normalize_data(self.lookup, self.objective_name)
        self.lookup = normalized_lookup

    @override
    def get_objective_value(self) -> float:
        """Get the objective value from the lookup table."""
        if self.target_mode_to_eval == TargetMode.MAX:
            MAX_NORMALIZED = 1.0
            return MAX_NORMALIZED
        if self.target_mode_to_eval == TargetMode.MIN:
            MIN_NORMALIZED = 0.0
            return MIN_NORMALIZED
        if self.target_mode_to_eval == TargetMode.MATCH and self.bounds is not None:
            raise NotImplementedError("Matching target mode not yet supported.")
        raise ValueError(f"Unrecognized target mode: '{self.target_mode_to_eval}'.")

    def _threshold_met(self, value: float) -> bool:
        """Check if the threshold is met."""
        if self.threshold is None:
            return True
        return abs(value) <= self.threshold

    @override
    def evaluate(self, prediction: DataFrame) -> float:
        """Evaluate the benchmarking metric and return the result."""
        target = self.get_objective_value()
        performance_row = f"{self.objective_name}_CumBest"
        normalized_prediction = self._normalize_data(prediction, performance_row)
        rooted_mean_squared_error = (
            (normalized_prediction[performance_row] - target) ** 2
        ).mean() ** 0.5
        if not self._threshold_met(rooted_mean_squared_error):
            raise AssertionError(
                f"Rooted mean squared error of {rooted_mean_squared_error} "
                f"exceeds threshold of {self.threshold}."
            )
        return -rooted_mean_squared_error

    @override
    def __str__(self) -> str:
        """Return the string representation of the metric."""
        return "Normalized Negative Root Mean Squared Error"
