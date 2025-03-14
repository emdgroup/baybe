"""Streamlit app to compare benchmarking results."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from baybe.serialization.core import _structure_dataframe_hook


def generate_compare_plot(
    version_a: pd.DataFrame, version_b: pd.DataFrame, figure_size: tuple = (15, 5)
) -> plt.Figure:
    """Generate a comparison plot between two benchmarks."""
    fig, ax = plt.subplots(1, 2, figsize=figure_size, sharey=True, sharex=True)

    cumbest_row_index = version_a.columns[version_a.columns.str.endswith("_CumBest")]
    cumbest_row_title = cumbest_row_index[0]

    sns.lineplot(
        data=version_a,
        marker="o",
        markersize=10,
        x="Iteration",
        y=cumbest_row_title,
        hue="Scenario",
        ax=ax[0],
    )

    sns.lineplot(
        data=version_b,
        marker="o",
        markersize=10,
        x="Iteration",
        y=cumbest_row_title,
        hue="Scenario",
        ax=ax[1],
    )

    x_scale_max = max(version_a["Iteration"].max(), version_b["Iteration"].max())
    x_scale_min = min(version_a["Iteration"].min(), version_b["Iteration"].min())

    ax[0].set_xlim(x_scale_min, x_scale_max)
    ax[1].set_xlim(x_scale_min, x_scale_max)

    return fig


@st.cache_data
def load_result(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load a benchmarking result from a file."""
    with open(path) as file:
        data = json.load(file)
    encoded_data = data.pop("data")
    dataframe = _structure_dataframe_hook(encoded_data, pd.DataFrame)
    return dataframe, data["metadata"]


def load_data(folder_path: Path) -> dict[str, tuple[pd.DataFrame, dict[str, str]]]:
    """Load benchmarking results from a folder."""
    results = {}
    for file_path in folder_path.iterdir():
        try:
            if file_path.suffix == ".json":
                result = load_result(file_path)
                filename = file_path.stem
                results[filename] = result
        except Exception as e:
            logging.error(f"Failed to load {file_path}: {e}")
    return results


@st.fragment
def main():
    """Run the benchmarking results app."""
    st.title("Benchmarking Results")

    folder_path = st.file_uploader(
        "Select a folder with benchmarking results", type="folder"
    )

    if folder_path is None:
        folder_path = Path(".")

    folder_path_class = Path(folder_path)
    results = load_data(folder_path_class)

    if not results:
        st.error("No benchmarking results found.")
        return

    st.title("Select two versions to compare:")
    left, right = st.columns(2)
    with left:
        version_a_key = st.selectbox("Left Plot", list(results.keys()))
    with right:
        version_b_key = st.selectbox("Right Plot", list(results.keys()))

    version_a, version_a_metainf = results[version_a_key]
    version_b, version_b_metainf = results[version_b_key]

    fig = generate_compare_plot(version_a, version_b)
    st.pyplot(fig)

    left, right = st.columns(2)
    with left:
        st.write("Metadata for version A")
        st.write(version_a_metainf)
    with right:
        st.write("Metadata for version B")
        st.write(version_b_metainf)


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Benchmarking Results", page_icon="📊")

    main()
