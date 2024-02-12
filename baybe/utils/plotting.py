"""Plotting utilities."""

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_plots(data: pd.DataFrame, name: str, **kwargs) -> None:
    """Create plots from a given data frame and save them as a svg file.

    The plots will be saved in the location in which the file that calls this function
    is being located. This method is intended to be used for plotting the results of the
    examples, but can also be used for other plots.

    If the ``SMOKE_TEST`` variable is set, no plots are being created and this method
    immediately returns.

    Using the ``BAYBE_MULTIVERSION_PLOTS`` environment variable, it is possible to
    create plots for the light and dark version of the documentation. If this variable
    is not set, a single file named `{name}_check.svg` is created.

    Note that it is necessary to provide keyword arguments for setting up the exact
    plot.

    Args:
        data: The data frame containing the data to be plotted.
        name: The name of the plot that should be created.
        **kwargs: Keyword arguments. Used for specifying the plot.
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
            path = Path(sys.path[0], f"{name}_{theme_name}.svg")
            plt.savefig(
                path,
                format="svg",
                transparent=True,
            )
            plt.clf()
        else:
            warnings.warn("No keyword arguments were provided when plotting.")
