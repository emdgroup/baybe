"""Quadratic transfer learning benchmark with different minima and few sources.

This benchmark tests transfer learning on quadratic functions where source and target
tasks have different minimum locations (b varies), using only 3 source tasks.
"""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.transfer_learning.quadratic.base import (
    get_optimal_target_value,
    quadratic_tl_convergence_benchmark,
)


def quadratic_diff_min_few_sources_tl(
    settings: ConvergenceBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark function for transfer learning with different minimum functions.

    Key characteristics:
    • Compares transfer learning vs. non-transfer learning approaches
    • Uses quadratic functions: y = a*(x+b)^2 + c
    • Configuration:
      - keep_min=False: Functions have different b values (different minimum locations)
      - n_sources=3: Uses 3 source tasks
    • Source tasks: 3 randomly generated quadratic functions with varying b ∈ [-1, 1]
    • Target task: 1 randomly generated quadratic function with varying b ∈ [-1, 1]
    • Tests varying amounts of source data:
      - 1% of source data
      - 2% of source data
      - 3% of source data
      - 5% of source data
      - 10% of source data
      - 20% of source data
    • Objective: Minimize quadratic function value
    • Optimal value: Theoretical minimum c value of target function

    This benchmark tests whether transfer learning helps when source and target tasks
    have different minimum locations, making knowledge transfer more challenging.

    Args:
        settings: Configuration settings for the convergence benchmark

    Returns:
        DataFrame containing benchmark results comparing TL vs non-TL performance
    """
    # Generate data to extract source task names
    from benchmarks.domains.transfer_learning.quadratic.base import load_data

    data = load_data(n_sources=3, keep_min=False)
    all_tasks = data["task"].unique()
    target_tasks = ["target"]
    source_tasks = [task for task in all_tasks if task != "target"]

    return quadratic_tl_convergence_benchmark(
        settings=settings,
        source_tasks=source_tasks,
        target_tasks=target_tasks,
        percentages=[0.01, 0.02, 0.03, 0.05, 0.10, 0.20],
        n_sources=3,
        keep_min=False,
    )


# Benchmark configuration
benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=5,
    n_mc_iterations=30,
)

# Calculate optimal target value
optimal_value = get_optimal_target_value()

# Create the benchmark
quadratic_diff_min_few_sources_tl_benchmark = ConvergenceBenchmark(
    function=quadratic_diff_min_few_sources_tl,
    optimal_target_values={"y": optimal_value},
    settings=benchmark_config,
)
