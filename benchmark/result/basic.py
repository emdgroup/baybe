"""Basci result classes for benchmarking."""

import pandas as pd
import seaborn as sns
from attrs import define
from matplotlib.figure import Figure, SubFigure
from pandas import DataFrame

from benchmark.result.base import Result


@define(frozen=True)
class SingleResult(Result):
    """A single result of the benchmarking."""

    result: DataFrame
    """The result of the benchmarking."""

    execution_time_ns: int
    """The execution time of the benchmark in nanoseconds."""

    def create_convergence_plot(self) -> Figure | SubFigure:
        """Generate a plot of the benchmarking result.

        See :func:`benchmark.result.base.Result.create_convergence_plot`
        for more information.
        """
        dataframe_index_titles = self.result.columns
        cumbest_col_name = dataframe_index_titles[-1]
        ax = sns.lineplot(
            data=self.result,
            marker="o",
            markersize=10,
            x="Num_Experiments",
            y=cumbest_col_name,
            hue="Scenario",
        )
        if ax.figure is None:
            raise ValueError("The plot could not be created.")
        return ax.figure


@define(frozen=True)
class MultiResult(Result):
    """A multi result of the benchmarking."""

    result: list[SingleResult]
    """The result of the benchmarking."""

    def get_mean_execution_time(
        self,
    ) -> float:
        """Return the mean execution time of the benchmarking."""
        return sum([result.execution_time_ns for result in self.result]) / len(
            self.result
        )

    def create_convergence_plot(self) -> Figure | SubFigure:
        """Generate a plot of the benchmarking result.

        Generates a plot that express the performance of all experiments
        by plotting the mean of all experiments. To get a plot of the
        best found value over the experiments, load the best found
        with the `get_best_found` method.
        See :func:`benchmark.result.base.Result.create_convergence_plot`
        for more information.
        """
        list_of_dataframes = [result.result for result in self.result]
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
        return max(self.result, key=lambda x: x.result.iloc[-1].max())
