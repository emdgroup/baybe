"""Basic error metric implementations for benchmarking."""

from attrs import define, field
from pandas import DataFrame
from typing_extensions import override

from baybe.targets import TargetMode
from src.metric.base import (
    GoalOrientedMetricInterface,
    Metric,
    NormalizationInterface,
)


@define
class NormalizedNegativeRootMeanSquaredErrorMetric(
    Metric, GoalOrientedMetricInterface, NormalizationInterface
):
    """Normalized negative root mean squared error metric for regression tasks."""

    lookup: DataFrame | tuple[float, float]
    """The lookup table or function to evaluate the goal orientation
    metric and compare the best included result."""

    objective_name: str
    """The name of the objective to evaluate."""

    target_mode_to_eval: TargetMode = field(converter=TargetMode)
    """The target mode of the optimization under evaluation."""

    threshold: float = field(default=None)
    """Optional threshold for the metric.
    If the metric is below the threshold, an exception is thrown."""

    _max_value: float = field(init=False)
    """The maximum value in the lookup table or function."""

    _min_value: float = field(init=False)
    """The minimum value in the lookup table or function."""

    @override
    def _normalize_data(self, data: DataFrame, index_name: str) -> DataFrame:
        """Normalize the specified column in the DataFrame using min-max normalization.

        Args:
            data: The input DataFrame containing the data to be normalized.
            index_name: The name of the column to be normalized.

        Returns:
            DataFrame: The DataFrame with the specified column normalized.
        """
        data[index_name] = data[index_name].apply(
            lambda x: (x - self._min_value) / (self._max_value - self._min_value)
        )
        return data

    def __attrs_post_init__(self) -> None:
        """Post-initialization method for the class.

        Returns:
            None

        Raises:
            ValueError: If `target_mode_to_eval` is `TargetMode.MATCH`.
        """
        if self.target_mode_to_eval == TargetMode.MATCH:
            raise ValueError("Matching target mode not yet supported.")
        LOOKUP_IS_MIN_MAX = isinstance(self.lookup, tuple)
        if LOOKUP_IS_MIN_MAX:
            self._min_value, self._max_value = self.lookup
            return
        cumbest_row = self.lookup.filter(like="CumBest").columns.tolist()[0]
        self._max_value = self.lookup[cumbest_row].max()
        self._min_value = self.lookup[cumbest_row].min()

    @override
    def get_objective_value(self) -> float:
        """Get the objective value from the lookup table based on the target mode.

        Returns:
            float: The objective value, which is 1.0 for TargetMode.MAX,
                   0.0 for TargetMode.MIN, or raises an error for TargetMode.MATCH.

        Raises:
            NotImplementedError: If the target mode is TargetMode.MATCH
                                 and bounds are provided.
            ValueError: If the target mode is unrecognized.
        """
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
        """Evaluate the prediction against the target objective value.

        This method calculates the rooted mean squared error (RMSE) between the
        normalized prediction and the target objective value. If the RMSE exceeds
        a predefined threshold, an AssertionError is raised.

        Args:
            prediction: The predicted values to be evaluated.

        Returns:
            float: The negative rooted mean squared error.

        Raises:
            AssertionError: If a threshold is set and the metric exceeds the threshold.
        """
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
