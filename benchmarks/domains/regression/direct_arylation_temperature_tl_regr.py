"""TL regression benchmark for direct arylation data with temperature as task."""

from __future__ import annotations

import pandas as pd

from baybe.parameters import (
    TaskParameter,
)
from baybe.searchspace import SearchSpace
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


def create_searchspaces(
    data: pd.DataFrame,
) -> tuple[SearchSpace, SearchSpace, str, list[str], str]:
    """Create search spaces for vanilla GP and transfer learning models."""
    # Create SearchSpace without task parameter (vanilla GP)
    vanilla_searchspace = make_searchspace(data=data, use_task_parameter=False)

    # Create transfer learning search space (with task parameter)
    tl_searchspace = make_searchspace(data=data, use_task_parameter=True)

    # Extract task parameter details
    task_param = next(
        p for p in tl_searchspace.parameters if isinstance(p, TaskParameter)
    )
    name_task = task_param.name
    target_task = task_param.active_values[0]  # Extract single target task
    all_values = task_param.values
    source_tasks = [val for val in all_values if val != target_task]

    return vanilla_searchspace, tl_searchspace, name_task, source_tasks, target_task


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
        create_searchspaces_fn=create_searchspaces,
        create_objective_fn=make_objective,
    )
    return results_df


# Define the benchmark settings
benchmark_config = TransferLearningRegressionBenchmarkSettings(
    n_mc_iterations=30,  # 30,  # 5,
    max_n_train_points=10,  # 10,  # 10,
    source_fractions=(0.01, 0.05, 0.1, 0.2),  # , 0.05, 0.1, 0.2],  # 0.5, 0.7, 0.9],
    noise_std=0.0,  # Not used for real data
)

# Create the benchmark
direct_arylation_temperature_tl_regr_benchmark = TransferLearningRegressionBenchmark(
    function=direct_arylation_temperature_tl_regr, settings=benchmark_config
)
