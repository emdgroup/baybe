"""Script to visualize TL convergence benchmark results organized by source fraction."""

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



def visualize_tl_convergence_per_source(json_file_path):
    """Create visualization organized by source fraction."""
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

    # Parse scenarios and organize data
    baseline_scenarios = {}  # All 0_* scenarios
    source_data = {}

    for scenario in df["Scenario"].unique():
        source_pct, suffix, is_baseline_naive = parse_simple_scenario(scenario)

        # Skip legacy scenarios
        if suffix in ["searchspace", "full_searchspace", "reduced_searchspace"]:
            continue
        if scenario == "0":  # Skip the phantom "0" scenario
            continue

        if source_pct == 0:
            # All 0_* scenarios are baselines
            baseline_scenarios[suffix] = scenario
        else:
            if source_pct not in source_data:
                source_data[source_pct] = {}
            source_data[source_pct][suffix] = scenario

    # Get sorted source percentages (excluding 0 since that's handled as baselines)
    source_percentages = sorted([pct for pct in source_data.keys() if pct > 0])
    print(f"Source percentages: {source_percentages}")
    print(f"Baseline scenarios: {list(baseline_scenarios.keys())}")

    # Get all unique suffixes across all source percentages
    all_suffixes = set()
    for suffix_dict in source_data.values():
        all_suffixes.update(suffix_dict.keys())
    all_suffixes = sorted(all_suffixes)
    print(f"TL Model suffixes: {all_suffixes}")

    # Set up the plot style
    plt.style.use("default")
    sns.set_palette("tab10")
    
    # Create figure with subplots: 1 row (CumBest only) x (1 baseline + n_source_pct) columns
    n_cols = 1 + len(source_percentages)  # 1 for baselines + one per source percentage
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=True)

    # Handle single column case
    if n_cols == 1:
        axes = np.array([axes])  # Ensure it's always an array
    
    fig.suptitle(
        f"TL Convergence by Source Fraction: {metadata['name']}", 
        fontsize=14, fontweight="bold"
    )

    # Define unified, extensible color scheme for TL suffixes
    suffix_palette = [
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

    # Dynamically assign colors to detected suffixes
    suffix_colors = {}
    for i, suffix in enumerate(sorted(all_suffixes)):
        color_idx = i % len(suffix_palette)
        suffix_colors[suffix] = suffix_palette[color_idx]

    # Define baseline naive style (grey dashed)
    baseline_naive_style = {"color": "#95a5a6", "linestyle": "--", "alpha": 0.8, "linewidth": 2}

    def plot_scenarios_on_axes(ax, scenarios_to_plot, title_text, baselines_to_include=None):
        """Helper function to plot scenarios on given axes."""
        # Plot baseline scenarios if requested
        if baselines_to_include is not None:
            for baseline_suffix, baseline_scenario in baselines_to_include.items():
                scenario_df = df[df["Scenario"] == baseline_scenario]
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

                    # Special styling for baseline naive (grey dashed)
                    if baseline_suffix == "naive":
                        style = baseline_naive_style
                        label = "0_naive"
                    else:
                        # Use same colors as the TL suffixes for consistency
                        style = {
                            "color": suffix_colors.get(baseline_suffix, "#999999"),
                            "linestyle": "-",
                            "alpha": 1.0,
                            "linewidth": 2
                        }
                        label = baseline_suffix

                    # Plot baseline scenario
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

        # Plot the transfer learning scenarios
        for suffix, scenario in scenarios_to_plot.items():
            scenario_df = df[df["Scenario"] == scenario]

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

            # Use suffix-based styling
            style = {
                "color": suffix_colors[suffix],
                "linestyle": "-",
                "alpha": 1.0,
                "linewidth": 2
            }
            label = suffix
            
            # Plot CumBest
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

        # No individual legends - unified legend will be created later

    # First column: All baseline scenarios (0% source data)
    ax_baseline = axes[0]
    plot_scenarios_on_axes(
        ax_baseline,
        {}, "0% Source Data", baselines_to_include=baseline_scenarios
    )

    # Remaining columns: Each source percentage with TL models + baseline naive only
    for col_idx, source_pct in enumerate(source_percentages, 1):
        suffixes_for_source = source_data[source_pct].copy()

        ax = axes[col_idx]

        # Only show baseline naive in other columns to avoid confusion
        baseline_naive_only = {k: v for k, v in baseline_scenarios.items() if k == "naive"}

        plot_scenarios_on_axes(
            ax,
            suffixes_for_source, f"{source_pct}% Source Data",
            baselines_to_include=baseline_naive_only
        )

    # Create unified legend in the first subplot (top-left) in lower right corner
    legend_elements = []

    # Add TL suffixes (solid lines) - these are the colored lines in columns 2-4
    for suffix in sorted(all_suffixes):
        if suffix in suffix_colors:
            color = suffix_colors[suffix]
            legend_elements.append(plt.Line2D([0], [0], color=color, linestyle='-',
                                            linewidth=2, label=suffix))

    # Add baseline naive (grey dashed line) - this is the reference line in all columns
    if "naive" in baseline_scenarios:
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
            input_path.stem.replace("_result", "_tl_convergence_per_source") + ".png"
        )
    else:
        output_path = input_path.parent / (
            input_path.stem + "_tl_convergence_per_source.png"
        )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"TL convergence per source visualization saved to: {output_path}")

    # Show basic statistics
    print("\nDataset statistics:")
    print(f"- Total rows: {len(df)}")
    print(f"- Scenarios: {len(df['Scenario'].unique())}")
    print(f"- Monte Carlo runs: {df['Monte_Carlo_Run'].nunique()}")
    print(f"- Iterations per run: {df['Iteration'].max() + 1}")

    plt.close()
    return output_path