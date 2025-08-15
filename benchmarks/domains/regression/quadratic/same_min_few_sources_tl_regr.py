"""Quadratic TL regression benchmark with same minimum and few sources.

This benchmark tests transfer learning regression performance on quadratic functions
where all source and target tasks have their minimum at the same location, using
few sources.
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


def quadratic_same_min_few_sources_tl_regr(
    settings: TransferLearningRegressionSettings,
) -> pd.DataFrame:
    """Regression benchmark for TL with quadratic functions and few sources.

    Key characteristics:
    • Compares transfer learning vs. vanilla GP regression performance
    • Uses quadratic functions: y = a*(x+b)^2 + c + noise
    • Configuration:
      - keep_min=True: All functions have b=0 (same minimum location at x=0)
      - n_sources=2: Uses 2 source tasks
    • Source tasks: 2 randomly generated quadratic functions with b=0
    • Target task: 1 randomly generated quadratic function with b=0
    • Evaluates regression metrics: RMSE, R2, MAE, etc.
    • Tests varying amounts of source data and target training points

    This benchmark tests whether transfer learning helps regression when source
    and target tasks have the same minimum location, making knowledge transfer
    easier.

    Args:
        settings: Configuration settings for the regression benchmark

    Returns:
        DataFrame with benchmark results
    """
    results_df = run_quadratic_tl_regression_benchmark(
        settings=settings,
        n_sources=2,  # Few sources (reduced to match working benchmarks)
        keep_min=True,  # Same minima
    )
    return results_df


# Create the benchmark
quadratic_same_min_few_sources_tl_regr_benchmark = TransferLearningRegression(
    function=quadratic_same_min_few_sources_tl_regr, settings=benchmark_config
)
