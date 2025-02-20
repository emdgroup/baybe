"""Benchmark on ArylHalides data with two distinct arylhalides as TL tasks."""

from __future__ import annotations

import pandas as pd

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import SubstanceParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)


def get_data() -> pd.DataFrame:
    """Load data for benchmark.

    Returns:
        Data for benchmark.
    """
    # TODO change path
    data_dir = (
        "/Users/karinhrovatin/Documents/code/" + "BayBE_benchmark/domains/ArylHalides/"
    )
    data = pd.read_table(data_dir + "data.csv", sep=",")
    data_raw = pd.read_table(data_dir + "data_raw.csv", sep=",")
    for substance in ["base", "ligand", "additive"]:
        data[substance + "_smiles"] = data[substance].map(
            dict(zip(data_raw[substance], data_raw[substance + "_smiles"]))
        )
    return data


data = get_data()

test_task = "1-iodo-4-methoxybenzene"
source_task = [
    # Dissimilar source task
    "1-chloro-4-(trifluoromethyl)benzene"
]


def space_data() -> (SingleTargetObjective, SearchSpace, pd.DataFrame, pd.DataFrame):
    """Definition of search space, objective, and data.

    Returns:
        Objective, search space, pre-measured task data (source task),
        and lookup for the active (target) task.
    """
    data_params = [
        SubstanceParameter(
            name=substance,
            data=dict(zip(data[substance], data[f"{substance}_smiles"])),
            encoding="MORDRED",
        )
        for substance in ["base", "ligand", "additive"]
    ]

    task_param = TaskParameter(
        name="aryl_halide",
        values=[test_task] + source_task,
        active_values=[test_task],
    )

    objective = SingleTargetObjective(NumericalTarget(name="yield", mode="MAX"))
    searchspace = SearchSpace.from_product(parameters=[*data_params, task_param])

    lookup = data.query(f'aryl_halide=="{test_task}"').copy(deep=True)
    initial_data = data.query("aryl_halide.isin(@source_task)", engine="python").copy(
        deep=True
    )

    return objective, searchspace, initial_data, lookup


def arylhalides_tl_substance(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Benchmark function comparing TL and non-TL campaigns.

    Inputs:
        base  Discrete substance with numerical encoding
        ligand  Discrete substance with numerical encoding
        additive    Discrete substance with numerical encoding
        Concentration   Continuous
        aryl_halide  Discrete task parameter
    Output: continuous
    Objective: Maximization
    Optimal Inputs: [
        {
            base    MTBD
            ligand  AdBrettPhos
            additive N,N-dibenzylisoxazol-3-amine
        }
    ]
    Optimal Output: 68.24812709999999
    """
    objective, searchspace, initial_data, lookup = space_data()

    campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
    )

    results = []
    for p in [0.01, 0.02, 0.05, 0.1, 0.2]:
        results.append(
            simulate_scenarios(
                {f"{int(100 * p)}": campaign},
                lookup,
                initial_data=[
                    initial_data.sample(frac=p) for _ in range(settings.n_mc_iterations)
                ],
                batch_size=settings.batch_size,
                n_doe_iterations=settings.n_doe_iterations,
                impute_mode="error",
            )
        )
    results = pd.concat(results)
    return results


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=10,
    n_mc_iterations=100,
)

arylhalides_tl_substance_benchmark = ConvergenceBenchmark(
    function=arylhalides_tl_substance,
    optimal_target_values={"yield": 68.24812709999999},
    settings=benchmark_config,
)
