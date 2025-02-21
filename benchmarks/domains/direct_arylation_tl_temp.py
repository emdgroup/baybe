"""Direct arylation with temperature as TL task, reproducing the paper."""

from __future__ import annotations

import os

import pandas as pd

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import (
    NumericalDiscreteParameter,
    SubstanceParameter,
    TaskParameter,
)
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from benchmarks.data.utils import DATA_PATH
from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)


def get_data() -> pd.DataFrame:
    """Load data for benchmark.

    Returns:
        Data for benchmark.
    """
    data_path = DATA_PATH + "DirectArylation" + os.sep
    data = pd.read_excel(data_path + "data.xlsx", index_col=0)
    data["Temp_C"] = data["Temp_C"].astype(str)
    return data


data = get_data()


def space_data() -> (
    SingleTargetObjective,
    SearchSpace,
    SearchSpace,
    pd.DataFrame,
    pd.DataFrame,
):
    """Definition of search space, objective, and data.

    Returns:
        Objective, TL search space, non-TL search space,
        pre-measured task data (source task),
        and lookup for the active (target) task.
    """
    data_params = [
        SubstanceParameter(
            name=substance,
            data=dict(zip(data[substance], data[f"{substance}_SMILES"])),
            # Instead of using RDKIT as in paper the
            # RDKIT2DDESCRIPTORS is used due to deprecation of
            # the former
            encoding="RDKIT2DDESCRIPTORS",
        )
        for substance in ["Solvent", "Base", "Ligand"]
    ] + [
        NumericalDiscreteParameter(
            name="Concentration",
            values=sorted(data["Concentration"].unique()),
            tolerance=0.001,
        ),
    ]
    task_param = TaskParameter(
        name="Temp_C",
        values=["90", "105", "120"],
        active_values=["105"],
    )

    objective = SingleTargetObjective(NumericalTarget(name="yield", mode="MAX"))
    searchspace = SearchSpace.from_product(parameters=[*data_params, task_param])
    searchspace_nontl = SearchSpace.from_product(parameters=data_params)

    lookup = data.query('Temp_C=="105"').copy(deep=True)
    initial_data = data.query('Temp_C!="105"').copy(deep=True)

    return objective, searchspace, searchspace_nontl, initial_data, lookup


def direct_arylation_tl_temp(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Benchmark function comparing TL and non-TL campaigns.

    Inputs:
        Solvent  Discrete substance with numerical encoding
        Base    Discrete substance with numerical encoding
        Ligand  Discrete substance with numerical encoding
        Concentration   Continuous
        Temp_C  Discrete task parameter
    Output: continuous
    Objective: Maximization
    Optimal Inputs: [
        {
            Base    Cesium acetate
            Ligand  SCHEMBL15068049
            Solvent DMAc
            Concentration   0.153
        },
        {
            Base    Cesium pivalate
            Ligand  SCHEMBL15068049
            Solvent DMAc
            Concentration   0.153
        },
    ]
    Optimal Output: 100.0
    """
    objective, searchspace, searchspace_nontl, initial_data, lookup = space_data()

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
    # No training data
    results.append(
        simulate_scenarios(
            {"0": campaign},
            lookup,
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            n_mc_iterations=settings.n_mc_iterations,
            impute_mode="error",
        )
    )
    # Non-TL campaign
    results.append(
        simulate_scenarios(
            {"non-TL": Campaign(searchspace=searchspace_nontl, objective=objective)},
            lookup,
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            n_mc_iterations=settings.n_mc_iterations,
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

direct_arylation_tl_temp_benchmark = ConvergenceBenchmark(
    function=direct_arylation_tl_temp,
    optimal_target_values={"yield": 100},
    settings=benchmark_config,
)
