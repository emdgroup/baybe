"""Plotting utilities."""

import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_plots(data: pd.DataFrame, path: Path, base_name: str, **kwargs) -> None:
    """Create plots from a given data frame and save them as a svg file.

    The plots will be saved in the location specified by ``path``.
    The attribute ``base_name`` is used to define the name of the outputs.

    If the ``SMOKE_TEST`` variable is set, no plots are being created and this method
    immediately returns.

    Using the ``BAYBE_MULTIVERSION_PLOTS`` environment variable, it is possible to
    create plots for the light and dark version of the documentation. If this variable
    is not set, a single file named ``{base_name}_check.svg`` is created.

    Args:
        data: The data frame containing the data to be plotted.
        path: The path to the directory in which the plots should be saved.
        base_name: The base name that is used for naming the output files.
        **kwargs: Keyword arguments. They are directly passed to ``sns.lineplot`` and
            are used for specifying the plot.
    """
    # Check whether we immediately return due to just running a SMOKE_TEST
    if "SMOKE_TEST" in os.environ:
        return

    # File containing all the themes
    themes = json.load(open(Path("plotting_themes.json")))
    # Environment variables for checking whether we want to have multiversion plots
    BAYBE_MULTIVERSION_PLOTS = "BAYBE_MULTIVERSION_PLOTS" in os.environ

    # Choose either all available or just the `check` theme.
    chosen_themes = themes if BAYBE_MULTIVERSION_PLOTS else ["check"]
    for theme_name in chosen_themes:
        # Extract and set the current theme
        theme = themes[theme_name]
        font_scale, rc_params = theme["font_scale"], theme["rc_params"]
        sns.set_theme(style="ticks", font_scale=font_scale, rc=rc_params)
        # Only if kwargs are being provided, a plot is actually created
        if kwargs:
            sns.lineplot(data=data, **kwargs)
            output_path = Path(path, f"{base_name}_{theme_name}.svg")
            plt.savefig(
                output_path,
                format="svg",
                transparent=True,
            )
            plt.clf()
        else:
            warnings.warn("No keyword arguments were provided when plotting.")
