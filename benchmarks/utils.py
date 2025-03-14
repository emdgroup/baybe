"""Utility functions for loading and comparing benchmarks."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from baybe.serialization.core import _structure_dataframe_hook


def load_benchmark_results(path: Path | str) -> pd.DataFrame:
    """Load a benchmarking result from a file.

    Args:
        path: The path to the benchmarking result file.

    Returns:
        The benchmarking result as a pandas DataFrame.
    """
    with open(path) as file:
        data = json.load(file)
    encoded_data = data.pop("data")
    dataframe = _structure_dataframe_hook(encoded_data, pd.DataFrame)
    return dataframe


def create_compare_plot(
    version_a: pd.DataFrame,
    version_b: pd.DataFrame | None = None,
    figure_size: tuple = (15, 5),
) -> plt.Figure:
    """Generate a comparison plot between two benchmarks.

    Args:
        version_a: The first benchmarking result.
        version_b: The second benchmarking result.
            If None, only the first result is plotted.
        figure_size: The size of the figure.

    Returns:
        The comparison plot as a matplotlib Figure.
    """
    if version_b is None:
        fig, ax = plt.subplots(1, 1, figsize=figure_size)
        ax = _generate_cumulative_best_plot(version_a, ax)
        plt.close(fig)
        return fig

    return _generate_comparison_subplot(version_a, version_b, figure_size)


def create_compare_plot_from_paths(
    path_a: Path | str, path_b: Path | str | None = None, figure_size: tuple = (15, 5)
) -> plt.Figure:
    """Load benchmarks from paths and generate a comparison plot.

    Args:
        path_a: The path to the first benchmarking result file.
        path_b: The path to the second benchmarking result file.
            If None, only the first result is plotted.
        figure_size: The size of the figure.

    Returns:
        The comparison plot as a matplotlib Figure.
    """
    version_a = load_benchmark_results(path_a)

    if path_b is None:
        return create_compare_plot(version_a, None, figure_size)

    version_b = load_benchmark_results(path_b)

    return create_compare_plot(version_a, version_b, figure_size)


def _generate_comparison_subplot(
    version_a: pd.DataFrame,
    version_b: pd.DataFrame,
    figure_size: tuple = (15, 5),
) -> plt.Figure:
    """Generate a comparison subplot.

    Args:
        version_a: The first benchmarking result.
        version_b: The second benchmarking result.
        figure_size: The size of the figure.

    Returns:
        The comparison subplot as a matplotlib Figure.
    """
    fig, ax = plt.subplots(1, 2, figsize=figure_size, sharey=True, sharex=True)

    ax[0] = _generate_cumulative_best_plot(version_a, ax[0])
    ax[1] = _generate_cumulative_best_plot(version_b, ax[1])

    x_scale_max = max(version_a["Iteration"].max(), version_b["Iteration"].max())
    x_scale_min = min(version_a["Iteration"].min(), version_b["Iteration"].min())

    ax[0].set_xlim(x_scale_min, x_scale_max)
    ax[1].set_xlim(x_scale_min, x_scale_max)

    plt.close(fig)

    return fig


def _generate_cumulative_best_plot(version_df: pd.DataFrame, ax: plt.Axes) -> plt.Axes:
    """Generate a cumulative best plot.

    Args:
        version_df: The benchmarking result DataFrame.
        ax: The matplotlib Axes object.

    Returns:
        The matplotlib Axes object with the cumulative best plot.
    """
    cumbest_row_index = version_df.columns[version_df.columns.str.endswith("_CumBest")]
    cumbest_row_title = cumbest_row_index[0]

    ax = sns.lineplot(
        data=version_df,
        marker="o",
        markersize=10,
        x="Iteration",
        y=cumbest_row_title,
        hue="Scenario",
        ax=ax,
    )

    return ax
