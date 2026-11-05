"""One-off case study: cost of discrete Cartesian-product construction strategies.

Compares six variants of BayBE's discrete Cartesian-product construction across
search space sizes and dataframe backends. Two builders are available:

- ``full``: build the full product, then apply filters.
- ``incremental``: cross-join parameters one at a time and filter as early as
  possible.

Both can optionally apply filters in canonical order (``sort_filters``), and the
incremental builder can optionally order parameters heuristically
(``sort_parameters``).

The scenario combines all discrete filtering constraints that implement a row-wise
expression (selection, repetition limit, linear, product, and cardinality), passed in
reverse canonical order so that sorting changes the filter sequence.

The measured quantity is the wall-clock time required to build and materialize the
candidate set. Results are written as a tidy CSV, per-backend pivot tables, a static
plot, and an interactive HTML plot.

Run with::

    python case_study_product_construction.py
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable, Sequence
from math import prod
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import narwhals.stable.v2 as nw
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from baybe._optional.info import POLARS_INSTALLED
from baybe.constraints import (
    DiscreteCardinalityConstraint,
    DiscreteLinearConstraint,
    DiscreteProductConstraint,
    DiscreteRepetitionLimitConstraint,
    DiscreteSelectionConstraint,
    SubSelectionCondition,
)
from baybe.constraints.base import DiscreteConstraint
from baybe.parameters import NumericalDiscreteParameter
from baybe.parameters.base import DiscreteParameter
from baybe.searchspace.utils import (
    _make_filtered_cartesian_product,
    _make_filtered_cartesian_product_incremental,
)
from baybe.settings import Settings

#: Variants under comparison, mapping a label to builder and construction flags.
VARIANTS: dict[str, tuple[Callable[..., Any], dict[str, bool]]] = {
    "full": (_make_filtered_cartesian_product, {}),
    "full+sorted-filters": (
        _make_filtered_cartesian_product,
        {"sort_filters": True},
    ),
    "incremental": (_make_filtered_cartesian_product_incremental, {}),
    "incremental+sorted-parameters": (
        _make_filtered_cartesian_product_incremental,
        {"sort_parameters": True},
    ),
    "incremental+sorted-filters": (
        _make_filtered_cartesian_product_incremental,
        {"sort_filters": True},
    ),
    "incremental+sorted-filters+parameters": (
        _make_filtered_cartesian_product_incremental,
        {"sort_filters": True, "sort_parameters": True},
    ),
}

#: Backends under comparison.
BACKENDS = ("pandas", "polars")

#: Color per variant, shared across backends and between both plot types.
VARIANT_COLORS = dict(zip(VARIANTS, qualitative.Plotly))


def _make_scenario(
    value_counts: Sequence[int],
) -> tuple[list[DiscreteParameter], list[DiscreteConstraint]]:
    """Create a search space scenario with the given parameter sizes.

    All constraints implementing a row-wise expression are included. They are passed
    in reverse canonical order so that enabling canonical sorting changes the
    effective filter sequence.

    Args:
        value_counts: The number of values of each parameter.

    Returns:
        The parameters and the filtering constraints of the scenario.
    """
    parameters: list[DiscreteParameter] = [
        NumericalDiscreteParameter(name=f"p{i}", values=list(range(n_values)))
        for i, n_values in enumerate(value_counts)
    ]
    names = [p.name for p in parameters]
    max_values = [n_values - 1 for n_values in value_counts]

    constraints: list[DiscreteConstraint] = [
        # Canonical position 5: supports early partial evaluation via the max bound
        DiscreteCardinalityConstraint(
            parameters=names,
            min_cardinality=2,
            max_cardinality=len(parameters) - 1,
        ),
        # Canonical position 4: requires all parameters
        DiscreteProductConstraint(
            parameters=names,
            operator="<=",
            rhs=0.01 * prod(max_values),
        ),
        # Canonical position 3: requires all parameters, weak pruning
        DiscreteLinearConstraint(
            parameters=names,
            operator="<=",
            rhs=0.8 * sum(max_values),
        ),
        # Canonical position 2: supports early partial evaluation
        DiscreteRepetitionLimitConstraint(
            parameters=names,
            n_max_repetitions=2,
        ),
        # Canonical position 1: strong, early pruning. Referencing the last
        # parameter makes the given (identity) order adversarially bad for
        # incremental construction, so the ordering heuristic has something to do.
        DiscreteSelectionConstraint(
            parameters=[names[0], names[-1]],
            conditions=[
                SubSelectionCondition(selection=[1, 2]),
                SubSelectionCondition(selection=[1, 2]),
            ],
            combiner="AND",
        ),
    ]
    return parameters, constraints


def _run_once(
    parameters: Sequence[DiscreteParameter],
    constraints: Sequence[DiscreteConstraint],
    backend: str,
    builder: Callable[..., Any],
    flags: dict[str, bool],
) -> tuple[float, int]:
    """Run a single construction and return its duration and candidate count.

    Args:
        parameters: The discrete parameters of the scenario.
        constraints: The filtering constraints of the scenario.
        backend: The dataframe backend to use.
        builder: The construction function to invoke.
        flags: The construction flags selecting the variant.

    Returns:
        The elapsed wall-clock time in seconds and the number of candidates.
    """
    start = time.perf_counter()
    with Settings(default_dataframe_backend=backend):
        result = builder(parameters, constraints, **flags)
        candidates = nw.from_native(result).lazy().collect()
    return time.perf_counter() - start, len(candidates)


def run_case_study(
    sizes: Sequence[Sequence[int]],
    repeats: int,
) -> pd.DataFrame:
    """Run all variants for all backends and search space sizes.

    Args:
        sizes: The parameter value counts defining the search space sizes.
        repeats: The number of timed repetitions per variant.

    Returns:
        A tidy dataframe with one row per backend, size, and variant.

    Raises:
        RuntimeError: If the variants produce different numbers of candidates.
    """
    backends = [b for b in BACKENDS if b != "polars" or POLARS_INSTALLED]
    rows: list[dict] = []

    for value_counts in sizes:
        parameters, constraints = _make_scenario(value_counts)
        n_parameters = len(value_counts)
        n_combinations = prod(value_counts)

        for backend in backends:
            samples: dict[str, list[float]] = {label: [] for label in VARIANTS}
            candidate_counts: dict[str, int] = {}

            # Warm-up
            for builder, flags in VARIANTS.values():
                _run_once(parameters, constraints, backend, builder, flags)

            # Interleave variants to even out transient machine effects
            for _ in range(repeats):
                for label, (builder, flags) in VARIANTS.items():
                    elapsed, n_candidates = _run_once(
                        parameters, constraints, backend, builder, flags
                    )
                    samples[label].append(elapsed)
                    candidate_counts[label] = n_candidates

            if len(set(candidate_counts.values())) != 1:
                raise RuntimeError(
                    f"Variants disagree for {n_parameters=}, {backend=}: "
                    f"{candidate_counts=}"
                )

            for label in VARIANTS:
                rows.append(
                    {
                        "backend": backend,
                        "variant": label,
                        "n_parameters": n_parameters,
                        "value_counts": ",".join(map(str, value_counts)),
                        "n_combinations": n_combinations,
                        "sort_parameters": flags.get("sort_parameters"),
                        "n_candidates": candidate_counts[label],
                        "median_seconds": statistics.median(samples[label]),
                        "min_seconds": min(samples[label]),
                    }
                )

    return pd.DataFrame(rows)


def _pivot(df: pd.DataFrame, backend: str) -> pd.DataFrame:
    """Create the per-backend pivot table for a tidy result dataframe.

    Args:
        df: The tidy result dataframe.
        backend: The backend to filter for.

    Returns:
        A pivot with search space sizes as rows and variants as columns.
    """
    return (
        df[df["backend"] == backend]
        .pivot(index="n_combinations", columns="variant", values="median_seconds")
        .reindex(columns=list(VARIANTS))
    )


def _write_static_plot(df: pd.DataFrame, path: Path) -> None:
    """Create and save the static comparison plot.

    Args:
        df: The tidy result dataframe.
        path: The output path of the plot.
    """
    backends = [b for b in BACKENDS if b in set(df["backend"])]
    fig, axes = plt.subplots(
        1, len(backends), figsize=(6.5 * len(backends), 5.5), sharey=True, squeeze=False
    )
    axes = axes[0]

    for ax, backend in zip(axes, backends, strict=True):
        data = df[df["backend"] == backend]
        for label in VARIANTS:
            subset = data[data["variant"] == label].sort_values("n_combinations")
            ax.plot(
                subset["n_combinations"],
                subset["median_seconds"],
                marker="o",
                color=VARIANT_COLORS[label],
                label=label,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(backend)
        ax.set_xlabel("raw parameter combinations")
        ax.grid(which="both", alpha=0.25)

    axes[0].set_ylabel("median build time [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _write_interactive_plot(df: pd.DataFrame, path: Path) -> None:
    """Create and save the interactive comparison plot.

    Args:
        df: The tidy result dataframe.
        path: The output path of the HTML plot.
    """
    backends = [b for b in BACKENDS if b in set(df["backend"])]
    fig = make_subplots(
        rows=1,
        cols=len(backends),
        subplot_titles=backends,
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )

    for column, backend in enumerate(backends, start=1):
        data = df[df["backend"] == backend]
        for label in VARIANTS:
            subset = data[data["variant"] == label].sort_values("n_combinations")
            fig.add_trace(
                go.Scatter(
                    x=subset["n_combinations"],
                    y=subset["median_seconds"],
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    showlegend=column == 1,
                    line={"color": VARIANT_COLORS[label]},
                    marker={"size": 8, "color": VARIANT_COLORS[label]},
                    customdata=subset[["n_candidates"]].to_numpy(),
                    hovertemplate=(
                        f"<b>{label}</b> · {backend}<br>"
                        "combinations: %{x:,}<br>"
                        "candidates: %{customdata[0]:,}<br>"
                        "median: %{y:.6f} s<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )

    fig.update_xaxes(type="log", title_text="raw parameter combinations")
    fig.update_yaxes(type="log", title_text="median build time [s]")
    fig.update_layout(
        title="Discrete Cartesian product construction: optimization case study",
        template="plotly_white",
        width=1300,
        height=560,
        hovermode="closest",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
            "title": {"text": "variant"},
        },
        margin={"l": 70, "r": 220, "t": 80, "b": 60},
    )
    fig.write_html(path, include_plotlyjs=True, full_html=True)


def main() -> None:
    """Run the case study and write its outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=str,
        nargs="+",
        default=[
            "5,5,5,5",
            "5,5,5,5,5,5",
            "5,5,5,5,5,5,5,5",
            "5,5,5,5,5,5,5,5,5",
            "5,5,5,5,5,5,5,5,10",
            "5,5,5,5,5,5,5,5,5,5",
            "5,5,5,5,5,5,5,5,5,10",
        ],
        help=(
            "Search space sizes, each given as a comma-separated list of the "
            "number of values per parameter."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repetitions per variant.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "case_study_output",
        help="Directory for the generated CSV, tables, and plots.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sizes = [[int(v) for v in spec.split(",")] for spec in args.sizes]
    df = run_case_study(sizes, args.repeats)

    df.to_csv(args.output_dir / "results.csv", index=False)

    tables = []
    for backend in BACKENDS:
        if backend not in set(df["backend"]):
            continue
        table = _pivot(df, backend)
        tables.append(f"=== {backend} (median seconds) ===\n{table.to_string()}")
    (args.output_dir / "tables.txt").write_text("\n\n".join(tables))
    print("\n\n".join(tables))

    _write_static_plot(df, args.output_dir / "comparison.png")
    _write_interactive_plot(df, args.output_dir / "comparison.html")
    print(f"\nResults written to: {args.output_dir}")


if __name__ == "__main__":
    main()
