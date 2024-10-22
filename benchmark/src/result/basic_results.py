"""Basic result classes for benchmarking."""

import logging

import pandas as pd
import seaborn as sns
from attrs import define
from matplotlib.figure import Figure, SubFigure
from pandas import DataFrame
from typing_extensions import override

from benchmark.src.metric import Metric
from benchmark.src.result.base import Result


@define(frozen=True)
class SingleResult(Result):
    """A single result of the benchmarking."""

    benchmark_result: DataFrame
    """The result of the benchmarking."""

    execution_time_ns: int
    """The execution time of the benchmark in nanoseconds."""

    @override
    def get_execution_time_ns(self) -> float:
        """Return the execution time of the benchmark in nanoseconds."""
        return self.execution_time_ns

    @override
    def create_convergence_plot(self) -> Figure | SubFigure:
        """Generate a plot of the benchmarking result.

        See :func:`benchmark.result.base.Result.create_convergence_plot`
        for more information.
        """
        dataframe_index_titles = self.benchmark_result.columns
        cumbest_col_name = dataframe_index_titles[-1]
        ax = sns.lineplot(
            data=self.benchmark_result,
            marker="o",
            markersize=10,
            x="Num_Experiments",
            y=cumbest_col_name,
            hue="Scenario",
        )
        if ax.figure is None:
            raise ValueError("The plot could not be created.")
        return ax.figure

    @override
    def evaluate_result(
        self, metric: Metric, objective_scenario: list[str]
    ) -> dict[str, float]:
        """Evaluate the benchmarking result using the given metric.

        See :func:`benchmark.result.base.Result.evaluate_result` for more information.
        """
        metric_value = metric.evaluate(self.benchmark_result, objective_scenario)
        print(f"Metric: {metric} - Value: {metric_value}")
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


@define(frozen=True)
class MultiResult(Result):
    """A multi result of the benchmarking."""

    benchmark_results: list[SingleResult]
    """The result of the benchmarking."""

    def get_mean_execution_time(
        self,
    ) -> float:
        """Return the mean execution time of the benchmarking."""
        return sum(
            [result.execution_time_ns for result in self.benchmark_results]
        ) / len(self.benchmark_results)

    @override
    def get_execution_time_ns(self) -> float:
        """Return the mean execution time of all benchmarks in nanoseconds.

        To get a execution time of the best found value over the experiments,
        load the best found with the `get_best_found` method.
        """
        return self.get_mean_execution_time()

    @override
    def create_convergence_plot(self) -> Figure | SubFigure:
        """Generate a plot of the benchmarking result.

        Generates a plot that express the performance of all experiments
        by plotting the mean of all experiments. To get a plot of the
        best found value over the experiments, load the best found
        with the `get_best_found` method.
        See :func:`benchmark.result.base.Result.create_convergence_plot`
        for more information.
        """
        list_of_dataframes = [
            result.benchmark_result for result in self.benchmark_results
        ]
        df_concat = pd.concat(list_of_dataframes)
        by_row_index = df_concat.groupby(df_concat.index)
        df_means = by_row_index.mean()
        dataframe_index_titles = df_means.columns
        cumbest_col_name = dataframe_index_titles[-1]
        ax = sns.lineplot(
            data=df_means,
            marker="o",
            markersize=10,
            x="Num_Experiments",
            y=cumbest_col_name,
            hue="Scenario",
        )
        if ax.figure is None:
            raise ValueError("The plot could not be created.")
        return ax.figure

    def get_best_found(self) -> SingleResult:
        """Return the best found result of the benchmarking."""
        return max(
            self.benchmark_results, key=lambda x: x.benchmark_result.iloc[-1].max()
        )

    @override
    def evaluate_result(
        self, metric: Metric, objective_scenario: list[str]
    ) -> dict[str, float]:
        """Evaluate the benchmarking result using the given metric.

        If a threshold is set a exception may be thrown if one result does not meet it.
        See :func:`benchmark.result.base.Result.evaluate_result` for more information.
        """
        metric_sum_dict = dict()
        for result in self.benchmark_results:
            metric_dict = metric.evaluate(result.benchmark_result, objective_scenario)
            for key, value in metric_dict.items():
                if key not in metric_sum_dict:
                    metric_sum_dict[key] = 0
                metric_sum_dict[key] += value
        metric_mean_dict = dict()
        for key, value in metric_sum_dict.items():
            metric_mean_dict[key] = value / len(self.benchmark_results)
            logging.info(
                f"Mean Metric: {metric} - for {key} with value: {metric_mean_dict[key]}"
            )
        return metric_mean_dict

    @override
    def to_csv(self, path: str | None = None) -> str | None:
        """Write all result to a csv file.

        Combine the results of all runs and write them to one csv file.
        If no path is given, the function will return the csv as a string.

        Args:
            path: Optional path to write the csv file to. If None, the function
                  will return the csv as a string.

        Returns:
            str | None: The csv as a string if no path is given.
        """
        list_of_dataframes = []
        iteration = 0
        for result in self.benchmark_results:
            iteration += 1
            result.benchmark_result["Iteration"] = iteration
            list_of_dataframes.append(result.benchmark_result)
        combined_results = pd.concat(list_of_dataframes, ignore_index=True)
        if path is None:
            return combined_results.to_csv(index=False)
        combined_results.to_csv(path, index=False)
