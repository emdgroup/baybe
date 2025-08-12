"""Quadratic TL regression benchmark with different minima and many sources.

This benchmark tests transfer learning regression performance on quadratic functions
where source and target tasks have different minimum locations, using many source tasks.
"""

from __future__ import annotations

import pandas as pd

from benchmarks.definition.regression import (
    TransferLearningRegression,
    TransferLearningRegressionSettings,
)
from benchmarks.domains.regression.base import generate_result_filename
from benchmarks.domains.regression.quadratic.base import (
    benchmark_config,
    run_quadratic_tl_regression_benchmark,
)


def quadratic_diff_min_many_sources_tl_regr(
    settings: TransferLearningRegressionSettings,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Regression benchmark for TL with different quadratic functions and many sources.

    Key characteristics:
    • Compares transfer learning vs. vanilla GP regression performance
    • Uses quadratic functions: y = a*(x+b)^2 + c + noise
    • Configuration:
      - keep_min=False: Functions have different b values (different minimum locations)
      - n_sources=5: Uses 5 source tasks (many sources)
    • Source tasks: 5 randomly generated quadratic functions with varying b ∈ [-1, 1]
    • Target task: 1 randomly generated quadratic function with varying b ∈ [-1, 1]
    • Evaluates regression metrics: RMSE, R2, MAE, etc.
    • Tests varying amounts of source data and target training points

    This benchmark tests whether having more source tasks with different minima
    improves transfer learning regression performance compared to fewer source tasks.

    Args:
        settings: Configuration settings for the regression benchmark

    Returns:
        Tuple containing:
        - DataFrame with benchmark results
        - List of metric names used
        - List of model names used
    """
    return run_quadratic_tl_regression_benchmark(
        settings=settings,
        n_sources=5,  # Many sources (reduced but still more than few)
        keep_min=False,  # Different minima
    )


# Create the benchmark
quadratic_diff_min_many_sources_tl_regr_benchmark = TransferLearningRegression(
    function=quadratic_diff_min_many_sources_tl_regr, settings=benchmark_config
)


if __name__ == "__main__":
    """Run the quadratic different minimum many sources TL regression benchmark."""
    import pandas as pd

    from benchmarks.domains.regression.base import generate_result_filename

    print("Running Transfer Learning Regression Benchmark...")

    # Run the benchmark
    results_df, metrics, model_names = quadratic_diff_min_many_sources_tl_regr(
        benchmark_config
    )

    # Generate filename and save results
    filename = generate_result_filename("quadratic_diff_min_many_sources_tl_regression")
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
