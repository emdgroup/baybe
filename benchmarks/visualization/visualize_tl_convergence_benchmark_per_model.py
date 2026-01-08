"""Script to visualize TL convergence benchmark results organized by model type."""

import base64
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def parse_simple_scenario(scenario_name):
    """Parse scenario name into source percentage and suffix.

    Expected format: {percentage}_{suffix}
    Returns: (source_pct, suffix, is_baseline_naive)
    """
    parts = scenario_name.split('_')
    if len(parts) < 2:
        return 0, scenario_name, False

    try:
        source_pct = int(parts[0])
        suffix = '_'.join(parts[1:])  # Handle multi-part suffixes
        is_baseline_naive = (source_pct == 0 and suffix == 'naive')
        return source_pct, suffix, is_baseline_naive
    except ValueError:
        return 0, scenario_name, False


def load_benchmark_data(json_file_path):
    """Load and decode the data field from a benchmark result JSON file."""
    with open(json_file_path) as f:
        result = json.load(f)

    # Decode the base64 encoded pickled DataFrame
    data_str = result["data"]
    pickled_df = base64.b64decode(data_str.encode("utf-8"))
    df = pickle.loads(pickled_df)

    # Create unified Monte_Carlo_Run column
    # For scenarios with initial data: use Initial_Data as MC run index
    # For scenarios without initial data: convert Random_Seed to 0-based index
    df = df.copy()

    def create_mc_run(row):
        if not pd.isna(row['Initial_Data']):
            # Scenarios with initial data (1%, 5%, 10%): use Initial_Data
            return int(row['Initial_Data'])
        else:
            # Scenarios without initial data (0%): convert Random_Seed to 0-based index
            # Find minimum Random_Seed for this scenario to create 0-based index
            scenario_mask = df['Scenario'] == row['Scenario']
            min_seed = df.loc[scenario_mask, 'Random_Seed'].min()
            return int(row['Random_Seed'] - min_seed)

    df['Monte_Carlo_Run'] = df.apply(create_mc_run, axis=1)

    return df, result




def visualize_tl_convergence_per_model(json_file_path):
    """Create visualization organized by model type."""
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

    # Handle both 'scenario' and 'Scenario' column names
    scenario_col = 'scenario' if 'scenario' in df.columns else 'Scenario' if 'Scenario' in df.columns else None
    if scenario_col is None:
        raise ValueError(f"Could not find scenario column in DataFrame. Available columns: {list(df.columns)}")

    # Parse scenarios and organize data
    baseline_naive_scenario = None
    suffix_data = {}  # suffix -> {source_pct: scenario}

    for scenario in df[scenario_col].unique():
        source_pct, suffix, is_baseline_naive = parse_simple_scenario(scenario)

        # Skip legacy scenarios
        if suffix in ["searchspace", "full_searchspace", "reduced_searchspace"]:
            continue
        if scenario == "0":  # Skip the phantom "0" scenario
            continue

        if is_baseline_naive:
            baseline_naive_scenario = scenario
        else:
            if suffix not in suffix_data:
                suffix_data[suffix] = {}
            suffix_data[suffix][source_pct] = scenario

    # Get all unique suffixes
    available_suffixes = sorted(suffix_data.keys())
    print(f"Available suffixes: {available_suffixes}")
    print(f"Baseline naive scenario: {baseline_naive_scenario}")

    # Get all source percentages (including 0 for non-naive scenarios)
    all_source_pcts = set()
    for source_dict in suffix_data.values():
        all_source_pcts.update(source_dict.keys())
    all_source_pcts = sorted(all_source_pcts)
    print(f"Source percentages: {all_source_pcts}")

    # Set up the plot style
    plt.style.use("default")
    sns.set_palette("tab10")
    
    # Create figure with subplots: 1 row (CumBest only) x (1 baseline_naive + n_suffixes) columns
    n_cols = 1 + len(available_suffixes)  # 1 for baseline_naive + one per suffix
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=True)

    # Handle single column case
    if n_cols == 1:
        axes = np.array([axes])  # Ensure it's always an array

    fig.suptitle(
        f"TL Convergence by Model Type: {metadata['name']}",
        fontsize=14, fontweight="bold"
    )

    # Define unified, extensible color scheme for source fractions
    source_fraction_palette = [
        '#3498db',  # Bright blue
        '#e74c3c',  # Bright red
        '#2ecc71',  # Bright green
        '#f39c12',  # Bright orange
        '#9b59b6',  # Bright purple
        '#34495e',  # Dark blue-gray
        '#e67e22',  # Darker orange
        '#1abc9c',  # Teal
        '#8e44ad',  # Dark purple
        '#2c3e50',  # Very dark blue
        '#16a085',  # Dark teal
        '#c0392b',  # Dark red
        '#27ae60',  # Dark green
        '#d35400',  # Dark orange
    ]

    # Dynamically assign colors to detected source percentages
    source_colors = {}
    for i, source_pct in enumerate(all_source_pcts):
        color_idx = i % len(source_fraction_palette)
        source_colors[source_pct] = source_fraction_palette[color_idx]

    # Define baseline naive style (grey dashed)
    baseline_naive_style = {"color": "#95a5a6", "linestyle": "--", "alpha": 0.8, "linewidth": 2}

    def plot_suffix_scenarios(ax, source_scenarios, title_text, include_baseline_naive=True):
        """Helper function to plot scenarios for a specific suffix."""
        # Always plot baseline naive for comparison (grey dashed)
        if include_baseline_naive and baseline_naive_scenario:
            scenario_df = df[df[scenario_col] == baseline_naive_scenario]
            if not scenario_df.empty:
                grouped = (
                    scenario_df.groupby(["Monte_Carlo_Run", "Iteration"])
                    .agg({cum_best_col: "mean"})
                    .reset_index()
                    .groupby("Iteration")
                    .agg({cum_best_col: ["mean", "std"]})
                    .reset_index()
                )

                # Flatten column names
                grouped.columns = [
                    "Iteration",
                    "CumBest_mean",
                    "CumBest_std",
                ]

                # Fill NaN std values with 0
                grouped = grouped.fillna(0)

                iterations = grouped["Iteration"]

                # Plot baseline naive
                ax.plot(
                    iterations,
                    grouped["CumBest_mean"],
                    label="0_naive",
                    **baseline_naive_style
                )
                ax.fill_between(
                    iterations,
                    grouped["CumBest_mean"] - grouped["CumBest_std"],
                    grouped["CumBest_mean"] + grouped["CumBest_std"],
                    color=baseline_naive_style["color"],
                    alpha=0.15,
                )

        # Plot scenarios for each source percentage
        for source_pct, scenario in source_scenarios.items():
            scenario_df = df[df[scenario_col] == scenario]

            # Group by Monte Carlo run and iteration, then calculate mean and std across MC runs
            grouped = (
                scenario_df.groupby(["Monte_Carlo_Run", "Iteration"])
                .agg({cum_best_col: "mean"})
                .reset_index()
                .groupby("Iteration")
                .agg({cum_best_col: ["mean", "std"]})
                .reset_index()
            )

            # Flatten column names
            grouped.columns = [
                "Iteration",
                "CumBest_mean",
                "CumBest_std",
            ]

            # Fill NaN std values with 0
            grouped = grouped.fillna(0)

            iterations = grouped["Iteration"]

            # Use source percentage color and label
            style = {
                "color": source_colors[source_pct],
                "linestyle": "-",
                "alpha": 1.0,
                "linewidth": 2
            }
            label = f"{source_pct}%"

            # Plot scenarios
            ax.plot(
                iterations,
                grouped["CumBest_mean"],
                label=label,
                **style
            )
            ax.fill_between(
                iterations,
                grouped["CumBest_mean"] - grouped["CumBest_std"],
                grouped["CumBest_mean"] + grouped["CumBest_std"],
                color=style["color"],
                alpha=0.15,
            )

        # Add optimal target line if available
        if (
            "optimal_target_values" in metadata
            and target_name in metadata["optimal_target_values"]
        ):
            optimal_value = metadata["optimal_target_values"][target_name]
            ax.axhline(
                y=optimal_value,
                color="red",
                linestyle=":",
                alpha=0.7,
                linewidth=1,
            )

        # Customize subplot
        ax.set_title(title_text, fontsize=11)
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel(f"{cum_best_col}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    # First column: Only baseline naive ("No task parameter/no source data")
    ax_baseline = axes[0]
    plot_suffix_scenarios(
        ax_baseline,
        {}, "No task parameter/no source data", include_baseline_naive=True
    )

    # Remaining columns: Each suffix with its source percentages + baseline naive
    for col_idx, suffix in enumerate(available_suffixes, 1):
        source_scenarios = suffix_data[suffix]

        ax = axes[col_idx]

        # For naive suffix, don't include baseline naive to avoid duplicate
        include_baseline = (suffix != "naive")

        plot_suffix_scenarios(
            ax,
            source_scenarios, suffix, include_baseline_naive=include_baseline
        )
        
    # Create unified legend in the first subplot (top-left) in lower right corner
    legend_elements = []

    # Add source fractions (solid lines)
    for source_pct in sorted(all_source_pcts):
        if source_pct in source_colors:
            color = source_colors[source_pct]
            legend_elements.append(plt.Line2D([0], [0], color=color, linestyle='-',
                                            linewidth=2, label=f'{source_pct}%'))

    # Add baseline naive (grey dashed line)
    if baseline_naive_scenario:
        legend_elements.append(plt.Line2D([0], [0],
                                        color=baseline_naive_style["color"],
                                        linestyle=baseline_naive_style["linestyle"],
                                        linewidth=baseline_naive_style["linewidth"],
                                        label="0_naive"))
    
    # Add the legend to the first subplot in lower right corner
    first_ax = axes[0]
    first_ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
                   frameon=True, fancybox=True, shadow=True)

    # Adjust layout and fix title positioning
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, wspace=0.15)

    # Save the plot
    input_path = Path(json_file_path)
    if input_path.stem.endswith("_result"):
        output_path = input_path.parent / (
            input_path.stem.replace("_result", "_tl_convergence_per_model") + ".png"
        )
    else:
        output_path = input_path.parent / (
            input_path.stem + "_tl_convergence_per_model.png"
        )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"TL convergence per model visualization saved to: {output_path}")

    # Show basic statistics
    print("\nDataset statistics:")
    print(f"- Total rows: {len(df)}")
    print(f"- Scenarios: {len(df[scenario_col].unique())}")
    print(f"- Monte Carlo runs: {df['Monte_Carlo_Run'].nunique()}")
    print(f"- Iterations per run: {df['Iteration'].max() + 1}")

    plt.close()
    return output_path