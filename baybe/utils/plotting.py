"""Plotting utilities."""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def create_example_plots(
    ax: Axes,
    path: Path,
    base_name: str,
) -> None:
    """Create plots from an Axes object and save them as a svg file.

    The plots will be saved in the location specified by ``path``.
    The attribute ``base_name`` is used to define the name of the outputs.

    If the ``SMOKE_TEST`` variable is set, no plots are being created and this method
    immediately returns.

    The function attempts to read the predefined themes from ``plotting_themes.json``.
    For each theme it finds, a file ``{base_name}_{theme}.svg`` is being created.
    If the file cannot be found, if the JSON cannot be loaded or if the JSON is not well
    configured, a fallback theme is used.

    Args:
        ax: The Axes object containing the figure that should be plotted.
        path: The path to the directory in which the plots should be saved.
        base_name: The base name that is used for naming the output files.
    """
    # Check whether we immediately return due to just running a SMOKE_TEST
    if "SMOKE_TEST" in os.environ:
        return

    # Define a fallback theme in case no configuration is found
    fallback: Dict[str, Any] = {
        "color": "black",
        "figsize": (24, 8),
        "fontsize": 22,
        "framealpha": 0.3,
    }

    # Try to find the plotting themes by backtracking
    # Get the absolute path of the current script
    script_path = Path(sys.path[0]).resolve()
    while (
        not Path(script_path / "plotting_themes.json").is_file()
        and script_path != script_path.parent
    ):
        script_path = script_path.parent
    if script_path == script_path.parent:
        warnings.warn("No themes for plotting found. A fallback theme is used.")
        themes = {"fallback": fallback}
    else:
        # Open the file containing all the themes
        # If we reach this point, we know that the file exists, so we try to load it.
        # If the file is no proper json, the fallback theme is used.
        try:
            themes = json.load(open(script_path / "plotting_themes.json"))
        except json.JSONDecodeError:
            warnings.warn(
                "The JSON containing the themes could not be loaded."
                "A fallback theme is used.",
                UserWarning,
            )
            themes = {"fallback": fallback}

    for theme_name in themes:
        # Get all of the values from the themes
        # TODO This can probably be generalized and improved later on such that the
        # keys fit the rc_params of matplotlib
        # TODO We might want to add a generalization here
        necessary_keys = ("color", "figsize", "fontsize", "framealpha")
        if not all(key in themes[theme_name] for key in necessary_keys):
            warnings.warn(
                "Provided theme does not contain the necessary keys."
                "Using a fallback theme instead.",
                UserWarning,
            )
            current_theme = fallback
        else:
            current_theme = themes[theme_name]
        color: str = current_theme["color"]
        figsize: Tuple[int, int] = current_theme["figsize"]
        fontsize: int = current_theme["fontsize"]
        framealpha: float = current_theme["framealpha"]

        # Adjust the axes of the plot
        for key in ax.spines.keys():
            ax.spines[key].set_color(color)
        ax.xaxis.label.set_color(color)
        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_color(color)
        ax.yaxis.label.set_fontsize(fontsize)

        # Adjust the size of the ax
        # mypy thinks that ax.figure might become None, hence the explicit ignore
        if isinstance(ax.figure, Figure):
            ax.figure.set_size_inches(*figsize)
        else:
            warnings.warn("Could not adjust size of plot due to it not being a Figure.")

        # Adjust the labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(color)
            label.set_fontsize(fontsize)

        # Adjust the legend
        legend = ax.get_legend()
        legend.get_frame().set_alpha(framealpha)
        legend.get_title().set_color(color)
        legend.get_title().set_fontsize(fontsize)
        for text in legend.get_texts():
            text.set_fontsize(fontsize)
            text.set_color(color)

        output_path = Path(path, f"{base_name}_{theme_name}.svg")
        # mypy thinks that ax.figure might become None, hence the explicit ignore
        if isinstance(ax.figure, Figure):
            ax.figure.savefig(
                output_path,
                format="svg",
                transparent=True,
            )
        else:
            warnings.warn("Plots could not be saved.")
        plt.close()
