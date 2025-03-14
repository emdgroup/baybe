"""Utility functions for loading and comparing benchmarks."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from baybe.serialization.core import _structure_dataframe_hook


def load_benchmark_results(path: Path | str) -> pd.DataFrame:
    """Load a benchmarking result from a file."""
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
    """Generate a comparison plot between two benchmarks."""
    if version_b is None:
        fig, ax = plt.subplots(1, 1, figsize=figure_size)
        ax = _generate_cumulative_best_plot(version_a, ax)
        plt.close(fig)
        return fig

    return _generate_comparison_subplot(version_a, version_b, figure_size)


def _generate_comparison_subplot(version_a, version_b, figure_size):
    """Generate a comparison subplot."""
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
    """Generate a cumulative best plot."""
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
