"""TL regression benchmarks for aryl halide reactions."""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    TransferLearningRegressionBenchmark,
    TransferLearningRegressionBenchmarkSettings,
)
from benchmarks.definition.base import RunMode
from benchmarks.definition.regression.core import run_tl_regression_benchmark
from benchmarks.domains.aryl_halides.core import (
    load_data,
    make_objective,
    make_searchspace,
)


def aryl_halide_CT_I_BM_tl_regr(
    settings: TransferLearningRegressionBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark function for comparing regression performance of GP vs TL models.

    This benchmark uses different aryl halide substrates as tasks:
    - Source tasks: 1-chloro-4-(trifluoromethyl)benzene + 2-iodopyridine
    - Target task: 1-iodo-4-methoxybenzene

    It trains models with varying amounts of source and target data, and evaluates
    their predictive performance on held-out target data.

    Key characteristics:
    • Parameters:
      - Base: Substance with MORDRED encoding
      - Ligand: Substance with MORDRED encoding
      - Additive: Substance with MORDRED encoding
      - aryl_halide: Task parameter
    • Target: Reaction yield (continuous)
    • Objective: Maximization
    • Compares TL models (SourcePrior, MHGP, SHGP, Index Kernel) vs vanilla GP

    Args:
        settings: The benchmark settings.

    Returns:
        DataFrame with benchmark results
    """

    def make_searchspace_wrapper(data: pd.DataFrame, use_task_parameter: bool):
        if use_task_parameter:
            return make_searchspace(
                data=data,
                source_tasks=["1-chloro-4-(trifluoromethyl)benzene", "2-iodopyridine"],
                target_tasks=["1-iodo-4-methoxybenzene"],
            )
        else:
            return make_searchspace(data=data)

    return run_tl_regression_benchmark(
        settings=settings,
        data_loader=load_data,
        searchspace_factory=make_searchspace_wrapper,
        objective_factory=make_objective,
    )


def aryl_halide_CT_IM_tl_regr(
    settings: TransferLearningRegressionBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark function for comparing regression performance of GP vs TL models.

    This benchmark uses different aryl halide substrates as tasks:
    - Source task: 1-chloro-4-(trifluoromethyl)benzene
    - Target task: 1-iodo-4-methoxybenzene

    It trains models with varying amounts of source and target data, and evaluates
    their predictive performance on held-out target data.

    Key characteristics:
    • Parameters:
      - Base: Substance with MORDRED encoding
      - Ligand: Substance with MORDRED encoding
      - Additive: Substance with MORDRED encoding
      - aryl_halide: Task parameter
    • Target: Reaction yield (continuous)
    • Objective: Maximization
    • Compares TL models (SourcePrior, MHGP, SHGP, Index Kernel) vs vanilla GP

    Args:
        settings: The benchmark settings.

    Returns:
        DataFrame with benchmark results
    """

    def make_searchspace_wrapper(data: pd.DataFrame, use_task_parameter: bool):
        if use_task_parameter:
            return make_searchspace(
                data=data,
                source_tasks=["1-chloro-4-(trifluoromethyl)benzene"],
                target_tasks=["1-iodo-4-methoxybenzene"],
            )
        else:
            return make_searchspace(data=data)

    return run_tl_regression_benchmark(
        settings=settings,
        data_loader=load_data,
        searchspace_factory=make_searchspace_wrapper,
        objective_factory=make_objective,
    )


def aryl_halide_IP_CP_tl_regr(
    settings: TransferLearningRegressionBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark function for comparing regression performance of GP vs TL models.

    This benchmark uses different aryl halide substrates as tasks:
    - Source task: 2-iodopyridine
    - Target task: 3-chloropyridine

    It trains models with varying amounts of source and target data, and evaluates
    their predictive performance on held-out target data.

    Key characteristics:
    • Parameters:
      - Base: Substance with MORDRED encoding
      - Ligand: Substance with MORDRED encoding
      - Additive: Substance with MORDRED encoding
      - aryl_halide: Task parameter
    • Target: Reaction yield (continuous)
    • Objective: Maximization
    • Compares TL models (SourcePrior, MHGP, SHGP, Index Kernel) vs vanilla GP

    Args:
        settings: The benchmark settings.

    Returns:
        DataFrame with benchmark results
    """

    def make_searchspace_wrapper(data: pd.DataFrame, use_task_parameter: bool):
        if use_task_parameter:
            return make_searchspace(
                data=data,
                source_tasks=["2-iodopyridine"],
                target_tasks=["3-chloropyridine"],
            )
        else:
            return make_searchspace(data=data)

    return run_tl_regression_benchmark(
        settings=settings,
        data_loader=load_data,
        searchspace_factory=make_searchspace_wrapper,
        objective_factory=make_objective,
    )


# Benchmark configurations
aryl_halide_sou_CT_benchmark_config = TransferLearningRegressionBenchmarkSettings(
    n_mc_iterations_settings={
        RunMode.DEFAULT: 50,
        RunMode.SMOKETEST: 2,
    },
    max_n_train_points_settings={
        RunMode.DEFAULT: 25,
        RunMode.SMOKETEST: 2,
    },
    source_fractions_settings={
        RunMode.DEFAULT: (0.01, 0.05, 0.1, 0.2),
        RunMode.SMOKETEST: (0.01,),
    },
    noise_std_settings={
        RunMode.DEFAULT: 0.0,
        RunMode.SMOKETEST: 0.0,
    },
)

aryl_halide_IP_CP_tl_benchmark_config = TransferLearningRegressionBenchmarkSettings(
    n_mc_iterations_settings={
        RunMode.DEFAULT: 60,
        RunMode.SMOKETEST: 2,
    },
    max_n_train_points_settings={
        RunMode.DEFAULT: 25,
        RunMode.SMOKETEST: 2,
    },
    source_fractions_settings={
        RunMode.DEFAULT: (0.01, 0.05, 0.1, 0.2),
        RunMode.SMOKETEST: (0.01,),
    },
    noise_std_settings={
        RunMode.DEFAULT: 0.0,
        RunMode.SMOKETEST: 0.0,
    },
)

# Create benchmarks
aryl_halide_CT_I_BM_tl_regr_benchmark = TransferLearningRegressionBenchmark(
    function=aryl_halide_CT_I_BM_tl_regr, settings=aryl_halide_sou_CT_benchmark_config
)

aryl_halide_CT_IM_tl_regr_benchmark = TransferLearningRegressionBenchmark(
    function=aryl_halide_CT_IM_tl_regr, settings=aryl_halide_sou_CT_benchmark_config
)

aryl_halide_IP_CP_tl_regr_benchmark = TransferLearningRegressionBenchmark(
    function=aryl_halide_IP_CP_tl_regr, settings=aryl_halide_IP_CP_tl_benchmark_config
)
