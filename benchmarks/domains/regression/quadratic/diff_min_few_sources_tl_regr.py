"""Quadratic TL regression benchmark with different minima and few sources.

This benchmark tests transfer learning regression performance on quadratic functions
where source and target tasks have different minimum locations, using only 3 source
tasks.
"""

from __future__ import annotations

import pandas as pd

from benchmarks.definition.regression import (
    TransferLearningRegression,
    TransferLearningRegressionSettings,
)
from benchmarks.domains.regression.quadratic.base import (
    benchmark_config,
    run_quadratic_tl_regression_benchmark,
)


def quadratic_diff_min_few_sources_tl_regr(
    settings: TransferLearningRegressionSettings,
) -> pd.DataFrame:
    """Regression benchmark for TL with different quadratic functions and few sources.

    Key characteristics:
    • Compares transfer learning vs. vanilla GP regression performance
    • Uses quadratic functions: y = a*(x+b)^2 + c + noise
    • Configuration:
      - keep_min=False: Functions have different b values (different minimum locations)
      - n_sources=2: Uses 2 source tasks
    • Source tasks: 2 randomly generated quadratic functions with varying b ∈ [-1, 1]
    • Target task: 1 randomly generated quadratic function with varying b ∈ [-1, 1]
    • Evaluates regression metrics: RMSE, R2, MAE, etc.
    • Tests varying amounts of source data and target training points

    Tests whether transfer learning helps regression when source and target
    tasks have different minimum locations, making knowledge transfer more challenging.

    Args:
        settings: Configuration settings for the regression benchmark

    Returns:
        DataFrame with benchmark results
    """
    results_df = run_quadratic_tl_regression_benchmark(
        settings=settings,
        n_sources=3,
        keep_min=False,  # Different minima
    )
    return results_df


# Create the benchmark
quadratic_diff_min_few_sources_tl_regr_benchmark = TransferLearningRegression(
    function=quadratic_diff_min_few_sources_tl_regr, settings=benchmark_config
)


if __name__ == "__main__":
    """Run the quadratic different minimum few sources TL regression benchmark."""
    import pandas as pd

    from benchmarks.domains.regression.base import generate_result_filename

    print("Transfer Learning Regression Benchmark...")

    # Run the benchmark
    results_df = quadratic_diff_min_few_sources_tl_regr(benchmark_config)

    # Extract metrics and model names from column names for local run
    from benchmarks.visualize_regression_results import (
        extract_metrics_and_models_from_data,
    )

    metrics, model_names = extract_metrics_and_models_from_data(results_df)

    # Generate filename and save results
    filename = generate_result_filename("quadratic_diff_min_few_sources_tl_regression")
    print(f"Saving results to: {filename}")
    results_df.to_csv(filename, index=False)

    print("Benchmark completed!")
    print(f"Results saved to: {filename}")
    print(f"Metrics evaluated: {metrics}")
    print(f"Models compared: {model_names}")
    print(f"Total experiments: {len(results_df)}")

    # Show summary statistics
    print("\nSummary Statistics:")
    for metric in metrics:
        for model in model_names:
            col_name = f"{model}_{metric.lower()}"
            if col_name in results_df.columns:
                mean_val = results_df[col_name].mean()
                std_val = results_df[col_name].std()
                print(f"{model} {metric}: {mean_val:.4f} ± {std_val:.4f}")
