"""Test reproducibility for Michalewicz transfer learning benchmark."""

import os

os.environ["BAYBE_PARALLEL_SIMULATION_RUNS"] = "FALSE"


import pandas as pd
from botorch.test_functions import Michalewicz
from pandas.testing import assert_frame_equal

from baybe import Campaign
from baybe.simulation import simulate_scenarios
from benchmarks.definition import ConvergenceBenchmark, ConvergenceBenchmarkSettings
from benchmarks.domains.transfer_learning.michalewicz.michalewicz_tl_continuous import (
    make_initial_data,
    make_objective,
    make_searchspace,
    wrap_function,
)


def michalewicz_tl_continuous(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Benchmark function for transfer learning with the Michalewicz function in 5D.

    Key characteristics:
    • Compares transfer learning vs. non-transfer learning approaches
    • Uses two versions of Michalewicz function:
      - Target: standard negated Michalewicz
      - Source: negated Michalewicz with added noise (noise_std=0.15)
    • Tests varying amounts of source data (1, 10, 50, 100 points)
    • Includes baseline with no transfer learning (0 points)
    • Creates two campaign types:
      - Transfer learning: with task parameter to distinguish data sources
      - Non-transfer learning (naive): without task parameter

    Args:
        settings: Configuration settings for the convergence benchmark

    Returns:
        DataFrame containing benchmark results for all test cases
    """
    functions = {
        "Source_Function": Michalewicz(dim=5, negate=True, noise_std=0.15),
        "Target_Function": Michalewicz(dim=5, negate=True),
    }
    searchspace_nontl = make_searchspace(use_task_parameter=False)
    searchspace_tl = make_searchspace(use_task_parameter=True)

    objective = make_objective()
    campaign_tl = Campaign(
        searchspace=searchspace_tl,
        objective=objective,
    )
    campaign_nontl = Campaign(
        searchspace=searchspace_nontl,
        objective=objective,
    )

    p = 62
    res = simulate_scenarios(
        {f"{p}": campaign_tl, f"{p}_naive": campaign_nontl},
        lambda x: wrap_function(functions["Target_Function"], "Target_Function", x),
        initial_data=[
            make_initial_data(
                functions["Source_Function"],
                "Source_Function",
                p,
                settings.random_seed + i,
            )
            for i in range(settings.n_mc_iterations)
        ],
        batch_size=settings.batch_size,
        n_doe_iterations=settings.n_doe_iterations,
        impute_mode="error",
        random_seed=settings.random_seed,
    )

    return res


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=3,
    n_mc_iterations=1,
)

michalewicz_tl_continuous_benchmark = ConvergenceBenchmark(
    function=michalewicz_tl_continuous,
    optimal_target_values={"Target": 4.687658},
    settings=benchmark_config,
)

result_1 = michalewicz_tl_continuous_benchmark()
result_2 = michalewicz_tl_continuous_benchmark()
assert_frame_equal(result_1.data, result_2.data)
