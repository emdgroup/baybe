"""Visualization module for regression benchmarks."""

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from benchmarks.definition.regression import METRICS_HIGHER_IS_BETTER

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")


def plot_results(
    results: pd.DataFrame,
    metrics: list[str],
    model_names: list[str],
    file_name_prefix: str = "transfer_learning_regression",
) -> None:
    """Plot regression metrics vs. number of training points for each model.

    Creates individual plots for each metric and a combined grid plot with all metrics.

    Args:
        results: DataFrame containing benchmark results.
        metrics: List of metrics to plot.
        model_names: List of model names used in the benchmark.
        file_name_prefix: Prefix for the output plot files.
    """
    # Separate vanilla from other models
    vanilla_name = "vanilla"
    tl_model_names = [name for name in model_names if name != vanilla_name]

    # Get unique source fractions
    fractions = sorted(results["fraction_source"].unique())

    # Create a figure for the combined grid plot
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)  # Max 3 plots per row
    n_rows = math.ceil(n_metrics / n_cols)

    # Create combined figure - make it large enough
    fig_combined = plt.figure(figsize=(6 * n_cols, 5 * n_rows))

    # Create plots for each metric
    for i, metric in enumerate(metrics):
        metric_lower = metric.lower()

        # Determine if higher is better
        higher_is_better = METRICS_HIGHER_IS_BETTER.get(metric, True)

        # Create individual figure for this metric
        plt.figure(figsize=(12, 8))

        # Plot vanilla GP (same for all source fractions)
        vanilla_col = f"{vanilla_name}_{metric_lower}"
        vanilla_data = results.groupby("n_train_pts")[vanilla_col].mean().reset_index()
        plt.plot(
            vanilla_data["n_train_pts"],
            vanilla_data[vanilla_col],
            "k-",
            linewidth=2,
            label="Vanilla GP",
        )

        # Plot TL models for each source fraction
        markers = ["o", "s", "^", "D", "v"]  # Different markers for different models
        for j, model_name in enumerate(tl_model_names):
            marker = markers[j % len(markers)]
            for fraction in fractions:
                fraction_data = results[results["fraction_source"] == fraction]
                col_name = f"{model_name}_{metric_lower}"

                if col_name in fraction_data.columns:
                    tl_data = (
                        fraction_data.groupby("n_train_pts")[col_name]
                        .mean()
                        .reset_index()
                    )
                    plt.plot(
                        tl_data["n_train_pts"],
                        tl_data[col_name],
                        marker=marker,
                        linewidth=1.5,
                        label=f"{model_name} (source={int(fraction * 100)}%)",
                    )

        plt.xlabel("Number of Target Training Points")
        better_text = "higher" if higher_is_better else "lower"
        plt.ylabel(f"{metric} ({better_text} is better)")
        plt.title(f"{metric}: Vanilla GP vs. Transfer Learning")

        # Handle legend - if there are many entries, place it outside
        if len(tl_model_names) * len(fractions) > 5:
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            plt.legend(loc="best")

        plt.grid(True)
        plt.tight_layout()

        # Save individual figure
        plt.savefig(
            f"{file_name_prefix}_{metric_lower}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Plot saved as '{file_name_prefix}_{metric_lower}.png'")

        # Now add subplot to the combined figure
        plt.figure(fig_combined.number)
        ax = fig_combined.add_subplot(n_rows, n_cols, i + 1)

        # Plot vanilla GP on the subplot
        ax.plot(
            vanilla_data["n_train_pts"],
            vanilla_data[vanilla_col],
            "k-",
            linewidth=2,
            label="Vanilla GP",
        )

        # Plot TL models on the subplot
        for j, model_name in enumerate(tl_model_names):
            marker = markers[j % len(markers)]
            for fraction in fractions:
                fraction_data = results[results["fraction_source"] == fraction]
                col_name = f"{model_name}_{metric_lower}"

                if col_name in fraction_data.columns:
                    tl_data = (
                        fraction_data.groupby("n_train_pts")[col_name]
                        .mean()
                        .reset_index()
                    )
                    ax.plot(
                        tl_data["n_train_pts"],
                        tl_data[col_name],
                        marker=marker,
                        linewidth=1.5,
                        label=f"{model_name} ({int(fraction * 100)}%)",
                    )

        ax.set_xlabel("Number of Target Training Points")
        ax.set_ylabel(f"{metric} ({better_text})")
        ax.set_title(f"{metric}")
        ax.grid(True)

        # Only add legend to the first subplot to save space
        if i == 0:
            ax.legend(loc="best", fontsize="small")

    # Adjust the combined figure layout
    fig_combined.tight_layout()

    # Save the combined figure
    plt.savefig(f"{file_name_prefix}_all_metrics.png", dpi=300, bbox_inches="tight")
    print(f"Combined plot saved as '{file_name_prefix}_all_metrics.png'")
