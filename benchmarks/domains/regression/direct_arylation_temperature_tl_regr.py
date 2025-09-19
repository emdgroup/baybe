"""TL regression benchmark for direct arylation data with temperature as task."""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    TransferLearningRegressionBenchmark,
    TransferLearningRegressionBenchmarkSettings,
)
from benchmarks.definition.base import RunMode
from benchmarks.definition.regression.core import run_tl_regression_benchmark
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
        data_loader=load_data,
        searchspace_factory=make_searchspace,
        objective_factory=make_objective,
    )
    return results_df


# Define the benchmark settings
benchmark_config = TransferLearningRegressionBenchmarkSettings(
    n_mc_iterations_settings={
        RunMode.DEFAULT: 30,
        RunMode.SMOKETEST: 2,
    },
    max_n_train_points_settings={
        RunMode.DEFAULT: 10,
        RunMode.SMOKETEST: 2,
    },
    source_fractions_settings={
        RunMode.DEFAULT: (0.01, 0.05, 0.1, 0.2),
        RunMode.SMOKETEST: (0.1,),
    },
    noise_std_settings={
        RunMode.DEFAULT: 0.0,
        RunMode.SMOKETEST: 0.0,
    },
)

# Create the benchmark
direct_arylation_temperature_tl_regr_benchmark = TransferLearningRegressionBenchmark(
    function=direct_arylation_temperature_tl_regr, settings=benchmark_config
)
