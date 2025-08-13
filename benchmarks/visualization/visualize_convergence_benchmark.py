"""Script to visualize benchmark results from JSON files."""

import base64
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_benchmark_data(json_file_path):
    """Load and decode the data field from a benchmark result JSON file."""
    with open(json_file_path) as f:
        result = json.load(f)

    # Decode the base64 encoded pickled DataFrame
    data_str = result["data"]
    pickled_df = base64.b64decode(data_str.encode("utf-8"))
    df = pickle.loads(pickled_df)

    return df, result


def extract_benchmark_info(df):
    """Extract detailed information about the benchmark configuration."""
    info = {"surrogates": [], "scenario_descriptions": {}}

    # Default surrogate information (BayBE uses GaussianProcess unless specified)
    info["surrogates"] = ["GaussianProcess (default)"]

    # Extract scenario information from DataFrame
    scenarios = df["Scenario"].unique()

    # Create interpretable scenario descriptions with generic labels
    for scenario in scenarios:
        if "_naive" in scenario:
            base_scenario = scenario.replace("_naive", "")
            if base_scenario == "0":
                info["scenario_descriptions"][scenario] = (
                    "No Transfer Learning (0% source data)"
                )
            elif base_scenario.isdigit():
                percentage = int(base_scenario)
                info["scenario_descriptions"][scenario] = (
                    f"No Transfer Learning ({percentage}% source data)"
                )
            else:
                info["scenario_descriptions"][scenario] = (
                    f"No Transfer Learning ({scenario})"
                )
        elif scenario == "0":
            info["scenario_descriptions"][scenario] = (
                "Transfer Learning (0% source data)"
            )
        elif scenario.isdigit():
            percentage = int(scenario)
            info["scenario_descriptions"][scenario] = (
                f"Transfer Learning ({percentage}% source data)"
            )
        else:
            info["scenario_descriptions"][scenario] = f"Transfer Learning ({scenario})"

    return info


def visualize_convergence_benchmark(json_file_path):
    """Create visualization for benchmark results."""
    # Load data
    df, metadata = load_benchmark_data(json_file_path)

    # Automatically detect target column names
    iter_best_col = None
    cum_best_col = None
    target_name = None

    for col in df.columns:
        if col.endswith("_IterBest"):
            iter_best_col = col
            target_name = col.replace("_IterBest", "")
        elif col.endswith("_CumBest"):
            cum_best_col = col

    if iter_best_col is None or cum_best_col is None:
        raise ValueError(
            f"Could not find IterBest and CumBest columns in {json_file_path}. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"Detected target columns: {iter_best_col}, {cum_best_col}")

    # Extract detailed benchmark information
    benchmark_info = extract_benchmark_info(df)

    # Set up the plot style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(
        f"Benchmark Results: {metadata['name']}", fontsize=16, fontweight="bold"
    )

    # Get unique scenarios and sort them for consistent ordering
    scenarios = sorted(
        df["Scenario"].unique(),
        key=lambda x: (
            # Sort by numeric value first, then by whether it's naive
            int(x.split("_")[0]) if x.split("_")[0].isdigit() else float("inf"),
            "_naive" in x,
        ),
    )

    # Define colors for each scenario type
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))

    # Plot for each scenario
    for i, scenario in enumerate(scenarios):
        scenario_data = df[df["Scenario"] == scenario]

        # Group by iteration and calculate mean and std across Monte Carlo runs
        grouped = (
            scenario_data.groupby("Iteration")
            .agg({iter_best_col: ["mean", "std"], cum_best_col: ["mean", "std"]})
            .reset_index()
        )

        # Flatten column names
        grouped.columns = [
            "Iteration",
            "IterBest_mean",
            "IterBest_std",
            "CumBest_mean",
            "CumBest_std",
        ]

        # Fill NaN std values with 0 (when there's only one Monte Carlo run)
        grouped = grouped.fillna(0)

        iterations = grouped["Iteration"]

        # Determine line style (solid for transfer learning, dashed for naive)
        linestyle = "--" if "_naive" in scenario else "-"
        alpha = 0.7 if "_naive" in scenario else 1.0

        # Get interpretable label
        legend_label = benchmark_info["scenario_descriptions"].get(
            scenario, f"Scenario {scenario}"
        )

        # Plot IterBest
        ax1.plot(
            iterations,
            grouped["IterBest_mean"],
            label=legend_label,
            color=colors[i],
            linestyle=linestyle,
            alpha=alpha,
            linewidth=2,
        )
        ax1.fill_between(
            iterations,
            grouped["IterBest_mean"] - grouped["IterBest_std"],
            grouped["IterBest_mean"] + grouped["IterBest_std"],
            color=colors[i],
            alpha=0.2,
        )

        # Plot CumBest
        ax2.plot(
            iterations,
            grouped["CumBest_mean"],
            label=legend_label,
            color=colors[i],
            linestyle=linestyle,
            alpha=alpha,
            linewidth=2,
        )
        ax2.fill_between(
            iterations,
            grouped["CumBest_mean"] - grouped["CumBest_std"],
            grouped["CumBest_mean"] + grouped["CumBest_std"],
            color=colors[i],
            alpha=0.2,
        )

    # Add optimal target line if available
    if (
        "optimal_target_values" in metadata
        and target_name in metadata["optimal_target_values"]
    ):
        optimal_value = metadata["optimal_target_values"][target_name]
        ax1.axhline(
            y=optimal_value,
            color="red",
            linestyle=":",
            alpha=0.7,
            label=f"Optimal target: {optimal_value:.3f}",
        )
        ax2.axhline(
            y=optimal_value,
            color="red",
            linestyle=":",
            alpha=0.7,
            label=f"Optimal target: {optimal_value:.3f}",
        )
    elif (
        "optimal_target_values" in metadata and "y" in metadata["optimal_target_values"]
    ):
        # Fallback for legacy 'y' naming
        optimal_value = metadata["optimal_target_values"]["y"]
        ax1.axhline(
            y=optimal_value,
            color="red",
            linestyle=":",
            alpha=0.7,
            label=f"Optimal target: {optimal_value:.3f}",
        )
        ax2.axhline(
            y=optimal_value,
            color="red",
            linestyle=":",
            alpha=0.7,
            label=f"Optimal target: {optimal_value:.3f}",
        )

    # Customize first subplot (IterBest)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(iter_best_col)
    ax1.set_title(f"Best Value per Iteration ({iter_best_col})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Customize second subplot (CumBest)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(cum_best_col)
    ax2.set_title(f"Cumulative Best Value ({cum_best_col})")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add benchmark info as text
    settings_text = f"Settings: {metadata['settings']}"
    surrogate_text = f"Surrogates: {', '.join(benchmark_info['surrogates'])}"
    benchmark_text = (
        "Benchmark: 3 sources (quadratics),{df['Monte_Carlo_Run'].nunique()} MC runs"
    )

    info_text = f"{settings_text}\n{surrogate_text}\n{benchmark_text}"
    fig.text(0.02, 0.02, info_text, fontsize=8, alpha=0.7, verticalalignment="bottom")

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Save the plot
    input_path = Path(json_file_path)
    output_path = input_path.parent / (
        input_path.stem.replace("_result", "_vis") + ".png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {output_path}")

    # Show basic statistics
    print("\nDataset statistics:")
    print(f"- Total rows: {len(df)}")
    print(f"- Scenarios: {len(scenarios)}")
    print(f"- Monte Carlo runs: {df['Monte_Carlo_Run'].nunique()}")
    print(f"- Iterations per run: {df['Iteration'].max() + 1}")

    return output_path
