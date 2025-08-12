#!/usr/bin/env python3
"""Script to visualize transfer learning regression benchmark results from CSV files."""

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Define whether higher values are better for each metric
METRICS_HIGHER_IS_BETTER = {
    "RMSE": False,
    "R2": True,
    "MAE": False,
    "MAX_ERROR": False,
    "EXPLAINED_VARIANCE": True,
    "LPD": True,
    "NLPD": False,
}

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")


def extract_metrics_and_models_from_csv(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Extract available metrics and model names from CSV columns.

    Args:
        df: DataFrame with regression benchmark results

    Returns:
        Tuple of (metrics, model_names)
    """
    # Find all metric columns (exclude metadata columns)
    metadata_cols = {
        "mc_iter",
        "n_train_pts",
        "fraction_source",
        "n_source_pts",
        "n_test_pts",
    }
    metric_cols = [col for col in df.columns if col not in metadata_cols]

    # Extract unique model names and metrics
    models_and_metrics = []
    for col in metric_cols:
        parts = col.rsplit("_", 1)  # Split from right to separate model from metric
        if len(parts) == 2:
            model, metric = parts
            models_and_metrics.append((model, metric.upper()))

    # Get unique models and metrics
    models = sorted(list({model for model, metric in models_and_metrics}))
    metrics = sorted(list({metric for model, metric in models_and_metrics}))

    return metrics, models


def plot_regression_results_from_csv(csv_file_path: str) -> None:
    """Plot regression metrics from CSV file and save combined PNG file.

    Args:
        csv_file_path: Path to the CSV file with regression benchmark results
    """
    # Load the CSV file
    df = pd.read_csv(csv_file_path)
    print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")

    # Extract metrics and model names from the data
    metrics, model_names = extract_metrics_and_models_from_csv(df)
    print(f"Found metrics: {metrics}")
    print(f"Found models: {model_names}")

    # Separate vanilla from other models
    vanilla_name = "vanilla"
    tl_model_names = [name for name in model_names if name != vanilla_name]

    if vanilla_name not in model_names:
        print(f"Warning: No vanilla baseline found in models: {model_names}")
        vanilla_name = None

    # Get unique source fractions
    fractions = sorted(df["fraction_source"].unique())
    fractions = [
        fraction.item() if isinstance(fraction, np.generic) else fraction
        for fraction in fractions
    ]
    print(f"Source fractions: {fractions}")

    # Determine output file prefix from input file
    input_path = Path(csv_file_path)
    file_name_prefix = input_path.stem.replace("_results", "")
    output_dir = input_path.parent

    # Create combined figure with all metrics
    n_metrics = len(metrics)
    if n_metrics > 0:
        n_cols = min(3, n_metrics)  # Max 3 plots per row
        n_rows = math.ceil(n_metrics / n_cols)

        fig_combined, axes = plt.subplots(
            n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows)
        )
        if n_metrics == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        else:
            # axes is already in the correct shape for multiple rows and columns
            pass

        for i, metric in enumerate(metrics):
            metric_lower = metric.lower()
            higher_is_better = METRICS_HIGHER_IS_BETTER.get(metric, True)
            better_text = "higher" if higher_is_better else "lower"

            row, col = divmod(i, n_cols)
            if n_rows == 1:
                ax = axes[0, col]
            else:
                ax = axes[row, col]

            # Plot vanilla GP on the subplot
            if vanilla_name:
                vanilla_col = f"{vanilla_name}_{metric_lower}"
                if vanilla_col in df.columns:
                    vanilla_stats = (
                        df.groupby("n_train_pts")[vanilla_col]
                        .agg(["mean", "std"])
                        .reset_index()
                    )
                    vanilla_stats["std"] = vanilla_stats["std"].fillna(0)

                    ax.plot(
                        vanilla_stats["n_train_pts"],
                        vanilla_stats["mean"],
                        "k-",
                        linewidth=2,
                        label="Vanilla GP" if i == 0 else "",
                    )

                    ax.fill_between(
                        vanilla_stats["n_train_pts"],
                        vanilla_stats["mean"] - vanilla_stats["std"],
                        vanilla_stats["mean"] + vanilla_stats["std"],
                        color="black",
                        alpha=0.2,
                    )

            # Plot TL models on the subplot
            subplot_color_idx = 0
            subplot_n_colors = len(fractions) * len(tl_model_names)
            subplot_colors = plt.cm.tab10(np.linspace(0, 1, max(subplot_n_colors, 10)))

            for j, model_name in enumerate(tl_model_names):
                for fraction in fractions:
                    fraction_data = df[df["fraction_source"] == fraction]
                    col_name = f"{model_name}_{metric_lower}"

                    if col_name in fraction_data.columns and len(fraction_data) > 0:
                        tl_stats = (
                            fraction_data.groupby("n_train_pts")[col_name]
                            .agg(["mean", "std"])
                            .reset_index()
                        )
                        tl_stats["std"] = tl_stats["std"].fillna(0)

                        line_color = subplot_colors[subplot_color_idx]
                        subplot_color_idx += 1

                        ax.plot(
                            tl_stats["n_train_pts"],
                            tl_stats["mean"],
                            color=line_color,
                            linewidth=1.5,
                            label=f"{model_name} ({int(fraction * 100)}%)"
                            if i == 0
                            else "",
                        )

                        ax.fill_between(
                            tl_stats["n_train_pts"],
                            tl_stats["mean"] - tl_stats["std"],
                            tl_stats["mean"] + tl_stats["std"],
                            color=line_color,
                            alpha=0.2,
                        )

            ax.set_xlabel("Number of Target Training Points")
            ax.set_ylabel(f"{metric} ({better_text})")
            ax.set_title(f"{metric}")
            ax.grid(True, alpha=0.3)

            # Adjust y-axis for subplot
            subplot_means = []
            if vanilla_name:
                vanilla_col = f"{vanilla_name}_{metric_lower}"
                if vanilla_col in df.columns:
                    vanilla_means = df.groupby("n_train_pts")[vanilla_col].mean()
                    subplot_means.extend(vanilla_means.values)

            for model_name in tl_model_names:
                for fraction in fractions:
                    fraction_data = df[df["fraction_source"] == fraction]
                    col_name = f"{model_name}_{metric_lower}"
                    if col_name in fraction_data.columns and len(fraction_data) > 0:
                        tl_means = fraction_data.groupby("n_train_pts")[col_name].mean()
                        subplot_means.extend(tl_means.values)

            if subplot_means:
                mean_min, mean_max = min(subplot_means), max(subplot_means)
                mean_range = mean_max - mean_min
                padding = mean_range * 0.2 if mean_range > 0 else abs(mean_max) * 0.1
                ax.set_ylim(mean_min - padding, mean_max + padding)

            # Only add legend to the first subplot to save space
            if i == 0:
                ax.legend(loc="best", fontsize="x-small")

        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            if n_rows == 1:
                ax = axes[0, col]
            else:
                ax = axes[row, col]
            ax.set_visible(False)

        plt.tight_layout()

        # Save the combined figure with new name
        combined_output = output_dir / f"{file_name_prefix}_regression_metrics.png"
        plt.savefig(combined_output, dpi=300, bbox_inches="tight")
        print(f"Regression metrics plot saved as '{combined_output}'")

        plt.close()


def main():
    """Run benchmark."""
    parser = argparse.ArgumentParser(
        description="Visualize TL regression benchmark results from CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_regression_results.py single_file.csv
  python visualize_regression_results.py --file-names file1.csv file2.csv file3.csv
  python visualize_regression_results.py --file-names *_results.csv
        """,
    )

    # Support both single file (positional) and multiple files (--file-names)
    parser.add_argument("file", nargs="?", help="Single CSV result file to visualize")
    parser.add_argument(
        "--file-names", nargs="+", help="List of CSV result files to visualize"
    )

    args = parser.parse_args()

    # Determine which files to process
    if args.file and args.file_names:
        print("Error: Cannot specify both positional file and --file-names option.")
        sys.exit(1)
    elif args.file:
        csv_file_paths = [args.file]
    elif args.file_names:
        csv_file_paths = args.file_names
    else:
        print("Error: Must specify either a file or use --file-names option.")
        parser.print_help()
        sys.exit(1)

    successful_files = []
    failed_files = []

    for csv_file_path in csv_file_paths:
        if not os.path.exists(csv_file_path):
            print(f"Error: File '{csv_file_path}' does not exist.")
            failed_files.append(csv_file_path)
            continue

        if not csv_file_path.endswith(".csv"):
            print(f"Error: File '{csv_file_path}' is not a CSV file.")
            failed_files.append(csv_file_path)
            continue

        try:
            plot_regression_results_from_csv(csv_file_path)
            print(
                f"Successfully created regression visualization for '{csv_file_path}'!"
            )
            successful_files.append(csv_file_path)
        except Exception as e:
            print(f"Error creating visualization for '{csv_file_path}': {e}")
            failed_files.append(csv_file_path)

    # Summary
    print("\nSummary:")
    print(f"  Successfully processed: {len(successful_files)} files")
    print(f"  Failed: {len(failed_files)} files")

    if failed_files:
        print(f"  Failed files: {', '.join(failed_files)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
