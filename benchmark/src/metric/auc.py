"""Basic metrics for evaluating the performance of optimization algorithms."""

import numpy as np
from attrs import define, field
from pandas import DataFrame
from typing_extensions import override

from benchmark.src.metric.base import (
    Metric,
    NormalizationInterface,
)


@define
class NormalizedAreaUnderTheCurve(Metric, NormalizationInterface):
    """Normalize the Area Under the Curve (AUC) metric."""

    lookup: DataFrame | tuple[float, float]
    """The lookup table or function to evaluate the goal orientation
    metric and compare the best included result."""

    objective_name: str
    """The name of the objective to evaluate."""

    doe_iteration_header: str = field(default="Num_Experiments")
    """The name of the column in the DataFrame that
    contains the number of iterations."""

    threshold: dict[str, float] = field(default=None)
    """Optional threshold for the metric.
    The string indicates the strategy which the threshold should be applied on.
    It is the name of the strategy in the first scenario dataframe row."""

    _max_value_y: float = field(init=False)
    """The maximum value in the lookup table or function."""

    _min_value_y: float = field(init=False)
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
            lambda x: (x - self._min_value_y) / (self._max_value_y - self._min_value_y)
        )
        return data

    def __attrs_post_init__(self) -> None:
        """Post-initialization method for the class."""
        LOOKUP_IS_MIN_MAX = isinstance(self.lookup, tuple) and len(self.lookup) == 2
        if LOOKUP_IS_MIN_MAX:
            self._min_value_y, self._max_value_y = self.lookup
            return
        if isinstance(self.lookup, DataFrame):
            self._max_value_y = self.lookup[self.objective_name].max()
            self._min_value_y = self.lookup[self.objective_name].min()
            return
        raise TypeError("lookup must be a DataFrame when not using min-max tuple.")

    @override
    def _check_threshold(self, values: dict[str, float]) -> None:
        """Check if the threshold is met."""
        if self.threshold is None:
            return
        for threshold_scenario, threshold_value in self.threshold.items():
            THRESHOLD_VIOLATED = (
                threshold_scenario in values
                and abs(values[threshold_scenario]) < threshold_value
            )
            if THRESHOLD_VIOLATED:
                raise ValueError(
                    f"Threshold {threshold_value} violated for {threshold_scenario}"
                )

    @override
    def evaluate(
        self, prediction: DataFrame, objective_scenarios: list[str] | None = None
    ) -> dict[str, float]:
        """Calculate the Area Under the Curve (AUC) for given prediction data.

        Parameters:
            prediction: Prediction data containing the scenarios and their values.
                        objective_scenarios: A list of objective scenarios to evaluate.
                        If None, all scenarios are evaluated.
            objective_scenarios: The scenario names to calculate the metric.
                                 must match the the defined names in the
                                 :func:`baybe.simulation.scenarios.simulate_scenarios`
                                 scenarios dict.

        Returns:
            dict[str, float]: A dictionary where keys are scenario names
            and values are the computed AUC values.
        """
        normalized_y_dataframe = self._normalize_data(prediction, self.objective_name)
        scenario_header = normalized_y_dataframe.columns[0]
        cumbest_header = normalized_y_dataframe.columns[-1]
        scenarios = normalized_y_dataframe.groupby(scenario_header)
        auc_dict = {}
        iteration_tuple_dict = {}
        LIST_FOR_Y_MEAN = 1
        LIST_FOR_ITERATION_NUMBER = 0
        for scenario_name, scenario_group in scenarios:
            SZENARIO_NOT_TO_CONSIDER = (
                objective_scenarios is not None
                and scenario_name not in objective_scenarios
            )

            if SZENARIO_NOT_TO_CONSIDER:
                continue
            for doe_iter, cumbest in scenario_group.groupby(self.doe_iteration_header):
                if scenario_name not in iteration_tuple_dict:
                    iteration_tuple_dict[scenario_name] = ([], [])
                iteration_tuple_dict[scenario_name][LIST_FOR_ITERATION_NUMBER].append(
                    doe_iter
                )
                iteration_tuple_dict[scenario_name][LIST_FOR_Y_MEAN].append(
                    cumbest[cumbest_header].mean()
                )

        for scenario_name, group in iteration_tuple_dict.items():
            x = group[LIST_FOR_ITERATION_NUMBER]
            y = group[LIST_FOR_Y_MEAN]

            auc_dict[scenario_name] = np.trapz(y, x) / (max(x) - min(x))
        self._check_threshold(auc_dict)
        return auc_dict

    @override
    def __str__(self) -> str:
        """Return the string representation of the metric."""
        return "Normalized Area Under the Curve"
