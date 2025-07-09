"""Direct arylation reaction benchmark (non-transfer learning)."""

from __future__ import annotations

import pandas as pd

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
)
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from benchmarks.data.utils import DATA_PATH
from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)


def direct_arylation_single_batch(
    settings: ConvergenceBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark function for direct arylation reaction."""
    data = pd.read_table(
        DATA_PATH / "direct_arylation" / "data.csv",
        sep=",",
        index_col=0,
    )
    non_substance_paramters = [
        NumericalDiscreteParameter(
            name="Concentration",
            values=sorted(data["Concentration"].unique()),
        ),
        NumericalDiscreteParameter(
            name="Temp_C",
            values=sorted(data["Temp_C"].unique()),
        ),
    ]
    catgorical_parameters = [
        CategoricalParameter(name=substance, values=data[substance].unique())
        for substance in ["Solvent", "Base", "Ligand"]
    ]
    rdkit_parameters = [
        SubstanceParameter(
            name=substance,
            data=dict(zip(data[substance], data[f"{substance}_SMILES"])),
            encoding="RDKIT2DDESCRIPTORS",
        )
        for substance in ["Solvent", "Base", "Ligand"]
    ]
    rdkit_parameters = [
        SubstanceParameter(
            name=substance,
            data=dict(zip(data[substance], data[f"{substance}_SMILES"])),
            encoding="MORDRED",
        )
        for substance in ["Solvent", "Base", "Ligand"]
    ]

    categorical_searchspace = SearchSpace.from_product(
        parameters=[*catgorical_parameters, *non_substance_paramters]
    )
    rdkit_searchspace = SearchSpace.from_product(
        parameters=[*rdkit_parameters, *non_substance_paramters]
    )
    mordred_searchspace = SearchSpace.from_product(
        parameters=[*rdkit_parameters, *non_substance_paramters]
    )
    target = NumericalTarget(name="yield")
    objective = SingleTargetObjective(target=target)

    scenarios: dict[str, Campaign] = {
        "Random Recommender": Campaign(
            searchspace=categorical_searchspace,
            recommender=RandomRecommender(),
            objective=objective,
        ),
        "Categorical": Campaign(
            searchspace=categorical_searchspace,
            objective=objective,
        ),
        "RDKIT": Campaign(
            searchspace=rdkit_searchspace,
            objective=objective,
        ),
        "Mordred": Campaign(
            searchspace=mordred_searchspace,
            objective=objective,
        ),
    }

    results_df = simulate_scenarios(
        scenarios,
        data,
        batch_size=settings.batch_size,
        n_doe_iterations=settings.n_doe_iterations,
        n_mc_iterations=settings.n_mc_iterations,
        impute_mode="error",
    )
    return results_df


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=1,
    n_doe_iterations=30,
    n_mc_iterations=100,
)

direct_arylation_single_batch_benchmark = ConvergenceBenchmark(
    function=direct_arylation_single_batch,
    optimal_target_values={"yield": 100},
    settings=benchmark_config,
)
