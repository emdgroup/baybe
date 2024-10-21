"""Backtesting transfer learning from BayBE's docs."""

from uuid import UUID

import numpy as np
import pandas as pd
from botorch.test_functions.synthetic import Hartmann

from baybe.campaign import Campaign
from baybe.objective import SingleTargetObjective
from baybe.parameters import (
    NumericalDiscreteParameter,
    TaskParameter,
)
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.simulation.transfer_learning import simulate_transfer_learning
from baybe.targets.numerical import NumericalTarget
from baybe.utils.botorch_wrapper import botorch_function_wrapper
from benchmark.src import SingleExecutionBenchmark


def backtestin_tranfer_learning() -> tuple[pd.DataFrame, dict[str, str]]:
    """Backtesting transfer learning from BayBE's docs."""
    batch_size = 3
    n_doe_iterations = 25
    n_mc_iterations = 50
    DIMENSION = 3
    POINTS_PER_DIM = 7

    metadata = {
        "DOE_iterations": str(n_doe_iterations),
        "batch_size": str(batch_size),
        "n_mc_iterations": str(n_mc_iterations),
        "dimension": str(DIMENSION),
        "POINTS_PER_DIM ": str(POINTS_PER_DIM),
    }
    objective = SingleTargetObjective(target=NumericalTarget(name="Target", mode="MIN"))

    BOUNDS = Hartmann(dim=DIMENSION).bounds

    discrete_params = [
        NumericalDiscreteParameter(
            name=f"x{d}",
            values=np.linspace(lower, upper, POINTS_PER_DIM),
        )
        for d, (lower, upper) in enumerate(BOUNDS.T)
    ]

    task_param = TaskParameter(
        name="Function",
        values=["Hartmann", "Shifted"],
    )

    parameters = [*discrete_params, task_param]
    searchspace = SearchSpace.from_product(parameters=parameters)

    def shifted_hartmann(*x: float) -> float:
        """Calculate a shifted, scaled and noisy variant of the Hartman function."""
        noised_hartmann = Hartmann(dim=DIMENSION, noise_std=0.15)
        return 2.5 * botorch_function_wrapper(noised_hartmann)(x) + 3.25

    test_functions = {
        "Hartmann": botorch_function_wrapper(Hartmann(dim=DIMENSION)),
        "Shifted": shifted_hartmann,
    }

    grid = np.meshgrid(*[p.values for p in discrete_params])

    lookups: dict[str, pd.DataFrame] = {}
    for function_name, function in test_functions.items():
        lookup = pd.DataFrame(
            {f"x{d}": grid_d.ravel() for d, grid_d in enumerate(grid)}
        )
        lookup["Target"] = tuple(lookup.apply(function, axis=1))
        lookup["Function"] = function_name
        lookups[function_name] = lookup
    lookup = pd.concat([lookups["Hartmann"], lookups["Shifted"]]).reset_index()

    campaign = Campaign(searchspace=searchspace, objective=objective)

    results = simulate_transfer_learning(
        campaign,
        lookup,
        batch_size=batch_size,
        n_doe_iterations=n_doe_iterations,
        n_mc_iterations=n_mc_iterations,
    )

    for func_name, function in test_functions.items():
        task_param = TaskParameter(
            name="Function", values=["Hartmann", "Shifted"], active_values=[func_name]
        )
        parameters = [*discrete_params, task_param]
        searchspace = SearchSpace.from_product(parameters=parameters)
        result_baseline = simulate_scenarios(
            {
                f"{func_name}_No_TL": Campaign(
                    searchspace=searchspace, objective=objective
                )
            },
            lookups[func_name],
            batch_size=batch_size,
            n_doe_iterations=n_doe_iterations,
            n_mc_iterations=n_mc_iterations,
        )

    results = pd.concat([results, result_baseline])
    return results, metadata


benchmark_transfer_learning_backtesting = SingleExecutionBenchmark(
    title="Backtesting transfer learning from BayBE's docs.",
    identifier=UUID("78a3b90a-57f1-4914-bca2-4b6d8b68cbbe"),
    benchmark_function=backtestin_tranfer_learning,
)
