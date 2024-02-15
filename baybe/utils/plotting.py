"""Plotting utilities."""

import json
import os
import sys
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

    The function attempts to read the predefined themes from ``plotting_themes.json``.
    For each theme it finds, a file ``{base_name}_{theme}.svg`` is being created.
    If the file cannot be found, a single fallback theme is used.

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

    # First, we see if we happen to find the plotting themes in the current folder.
    # This is e.g. the case if we convert the file to a jupyter notebook.
    try:
        themes = json.load(open("plotting_themes.json"))
    except FileNotFoundError:
        # Try to find the plotting themes by backtracking
        # Get the absolute path of the current script
        script_path = Path(sys.argv[0]).resolve()
        # Backtrack until either the "baybe" folder is found or backtracking is no
        # longer possible
        while not script_path.name == "baybe" and script_path != script_path.parent:
            script_path = script_path.parent
        if script_path == script_path.parent:
            warnings.warn("No themes for plotting found. A default theme is used.")
            themes = {"check": {"font_scale": 1.75, "rc_params": {}}}
        else:
            # Open the file containing all the themes
            # This currently still assumes that the file is in the repo folder
            themes = json.load(open(script_path / "plotting_themes.json"))

    for theme_name in themes:
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
