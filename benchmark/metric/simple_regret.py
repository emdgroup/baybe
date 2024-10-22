"""Adds the simple regret metric to the benchmarking suite."""

from attrs import define, field
from pandas import DataFrame
from typing_extensions import override

from benchmark.metric.base import Metric, NormalizationInterface


@define
class NormalizedSimpleRegret(Metric, NormalizationInterface):
    """The simple regret metric."""

    evaluation_range: tuple[float, float] = field()
    """The range from the lowest value to the
    best value which can be found for a observed scenario."""

    objective_name: str
    """The name of the objective and therefore row to evaluate."""

    @override
    def evaluate(
        self, prediction: DataFrame, objective_scenarios: list[str] | None = None
    ) -> dict[str, float]:
        """Evaluate the given predictions against the objective scenario.

        Args:
            prediction: The predictions to evaluate from the an benchmark result.

            objective_scenarios: The scenario names to calculate the metric.
                Must match the  defined names in the
                :func:`baybe.simulation.scenarios.simulate_scenarios`
                scenarios dict. Only the appearing names will be evaluated.
                If None all scenarios will be evaluated.

        Returns:
            A dictionary containing evaluation metrics as keys
            and their corresponding values.
        """
        metric_dict = dict()
        NORMALIZED_MAX = 1.0
        CUMBEST_ROW = next(
            col for col in prediction.columns if "cumbest" in col.lower()
        )
        normalized_dataframe = self._normalize_data(prediction, CUMBEST_ROW)
        scenario_header = normalized_dataframe.columns[0]
        grouped_dataframe = normalized_dataframe.groupby(scenario_header)

        for scenario, data in grouped_dataframe:
            SCENARIO_TO_EVALUATE = (
                objective_scenarios is None or scenario in objective_scenarios
            )
            if SCENARIO_TO_EVALUATE:
                metric_dict[scenario] = NORMALIZED_MAX - data[CUMBEST_ROW].max()

        return metric_dict

    @override
    def _normalize_data(self, data: DataFrame, index_name: str) -> DataFrame:
        """Normalize the given data to scale the metric to [0,1].

        Args:
            data: The data to be normalized.
            index_name: The name of the index to be used for normalization.

        Returns:
            DataFrame: The normalized data.
        """
        changeable_data = data.copy()
        min_value, max_value = self.evaluation_range
        changeable_data[index_name] = (data[index_name] - min_value) / (
            max_value - min_value
        )
        return changeable_data
