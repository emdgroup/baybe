"""Direct arylation with temperature as TL task, reproducing the paper."""

from __future__ import annotations

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
        "/Users/karinhrovatin/Documents/code/"
        + "BayBE_benchmark/domains/DirectArylation/"
    )
    data = pd.read_excel(data_dir + "data.xlsx", index_col=0)
    data["Temp_C"] = data["Temp_C"].astype(str)
    return data


data = get_data()


def optimization_space() -> (SingleTargetObjective, SearchSpace):
    """Definition of search space and objective.

    Returns:
        Objective and search space.
    """
    parameters = [
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
        TaskParameter(
            name="Temp_C",
            values=["90", "105", "120"],
            active_values=["105"],
        ),
    ]

    objective = SingleTargetObjective(NumericalTarget(name="yield", mode="MAX"))
    searchspace = SearchSpace.from_product(parameters=parameters)

    return objective, searchspace


def lookup() -> pd.DataFrame:
    """Get lookup for the active task.

    Returns:
        Active task lookup.
    """
    return data.query('Temp_C=="105"').copy(deep=True)


def initial_data() -> pd.DataFrame:
    """Get starting data for the pre-measured tasks.

    Returns:
        Pre-measured task data.
    """
    return data.query('Temp_C!="105"').copy(deep=True)


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
    objective, searchspace = optimization_space()

    campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
    )

    lookup_data = lookup()
    initial_samples = initial_data()

    results = []
    for p in [0.01, 0.02, 0.05, 0.1, 0.2]:
        results.append(
            simulate_scenarios(
                {f"{int(100 * p)}": campaign},
                lookup_data,
                initial_data=[
                    initial_samples.sample(frac=p)
                    for _ in range(settings.n_mc_iterations)
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

direct_arylation_tl_temp_benchmark = ConvergenceBenchmark(
    function=direct_arylation_tl_temp,
    optimal_target_values={"yield": 100},
    settings=benchmark_config,
)
