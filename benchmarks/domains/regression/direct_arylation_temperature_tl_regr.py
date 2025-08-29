"""TL regression benchmark for direct arylation data with temperature as task."""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    TransferLearningRegressionBenchmark,
    TransferLearningRegressionBenchmarkSettings,
)
from benchmarks.domains.regression.base import run_tl_regression_benchmark
from benchmarks.domains.transfer_learning.direct_arylation.temperature_tl import (
    load_data,
    make_objective,
    make_searchspace,
)


def direct_arylation_temperature_tl_regr(
    settings: TransferLearningRegressionBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark function for comparing regression performance of GP vs TL models.

    This benchmark uses reactions at different temperatures:
    - Source tasks: 90°C and 120°C
    - Target task: 105°C

    It trains models with varying amounts of source and target data, and evaluates
    their predictive performance on held-out target data.

    Args:
        settings: The benchmark settings.

    Returns:
        DataFrame with benchmark results
    """
    results_df = run_tl_regression_benchmark(
        settings=settings,
        load_data_fn=load_data,
        make_searchspace_fn=make_searchspace,
        create_objective_fn=make_objective,
    )
    return results_df


# Define the benchmark settings
benchmark_config = TransferLearningRegressionBenchmarkSettings(
    n_mc_iterations=30,
    max_n_train_points=10,
    source_fractions=(0.01, 0.05, 0.1, 0.2),
    noise_std=0.0,
)

# Create the benchmark
direct_arylation_temperature_tl_regr_benchmark = TransferLearningRegressionBenchmark(
    function=direct_arylation_temperature_tl_regr, settings=benchmark_config
)
