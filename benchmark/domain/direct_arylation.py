"""Direct Arylation Benchmark."""

from uuid import UUID

from pandas import DataFrame, read_csv
from src import SingleExecutionBenchmark

from baybe.campaign import Campaign
from baybe.objective import SingleTargetObjective
from baybe.parameters import (
    CategoricalEncoding,
    CategoricalParameter,
    NumericalDiscreteParameter,
    SubstanceEncoding,
)
from baybe.parameters.substance import SubstanceParameter
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget, TargetMode
from domain.utils import PATH_PREFIX


def direct_arylation() -> tuple[DataFrame, dict[str, str]]:
    """Direct Arylation Simulation maximum yield with Mordred."""
    base_dict = {
        "Potassium acetate": "O=C([O-])C.[K+]",
        "Potassium pivalate": "O=C([O-])C(C)(C)C.[K+]",
        "Cesium acetate": "O=C([O-])C.[Cs+]",
        "Cesium pivalate": "O=C([O-])C(C)(C)C.[Cs+]",
    }
    ligand_dict = {
        "BrettPhos": "CC(C)C1=CC(C(C)C)=C(C(C(C)C)=C1)C2=C(P(C3CCCCC3)C4CCCCC4)"
        + "C(OC)=CC=C2OC",
        "Di-tert-butylphenylphosphine": "CC(C)(C)P(C1=CC=CC=C1)C(C)(C)C",
        "(t-Bu)PhCPhos": "CN(C)C1=CC=CC(N(C)C)=C1C2=CC=CC=C2P(C(C)(C)C)C3=CC=CC=C3",
        "Tricyclohexylphosphine": "P(C1CCCCC1)(C2CCCCC2)C3CCCCC3",
        "PPh3": "P(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3",
        "XPhos": "CC(C1=C(C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C(C(C)C)=CC(C(C)C)=C1)C",
        "P(2-furyl)3": "P(C1=CC=CO1)(C2=CC=CO2)C3=CC=CO3",
        "Methyldiphenylphosphine": "CP(C1=CC=CC=C1)C2=CC=CC=C2",
        "1268824-69-6": "CC(OC1=C(P(C2CCCCC2)C3CCCCC3)C(OC(C)C)=CC=C1)C",
        "JackiePhos": "FC(F)(F)C1=CC(P(C2=C(C3=C(C(C)C)C=C(C(C)C)C=C3C(C)C)C(OC)"
        + "=CC=C2OC)C4=CC(C(F)(F)F)=CC(C(F)(F)F)=C4)=CC(C(F)(F)F)=C1",
        "SCHEMBL15068049": "C[C@]1(O2)O[C@](C[C@]2(C)P3C4=CC=CC=C4)(C)O[C@]3(C)C1",
        "Me2PPh": "CP(C)C1=CC=CC=C1",
    }
    solvent_dict = {
        "DMAc": "CC(N(C)C)=O",
        "Butyornitrile": "CCCC#N",
        "Butyl Ester": "CCCCOC(C)=O",
        "p-Xylene": "CC1=CC=C(C)C=C1",
    }

    direct_arylation_mordred = [
        SubstanceParameter(
            name="Base",
            encoding=SubstanceEncoding.MORDRED,
            data=base_dict,
        ),
        SubstanceParameter(
            name="Ligand",
            encoding=SubstanceEncoding.MORDRED,
            data=ligand_dict,
        ),
        SubstanceParameter(
            name="Solvent",
            encoding=SubstanceEncoding.MORDRED,
            data=solvent_dict,
        ),
        NumericalDiscreteParameter("Concentration", values=(0.057, 0.1, 0.153)),
        NumericalDiscreteParameter("Temp_C", values=(90.0, 105.0, 120.0)),
    ]

    direct_arylation_ohe = [
        CategoricalParameter(
            name="Base",
            encoding=CategoricalEncoding.OHE,
            values=tuple(base_dict.keys()),
        ),
        CategoricalParameter(
            name="Ligand",
            encoding=CategoricalEncoding.OHE,
            values=tuple(ligand_dict.keys()),
        ),
        CategoricalParameter(
            name="Solvent",
            encoding=CategoricalEncoding.OHE,
            values=tuple(solvent_dict.keys()),
        ),
        NumericalDiscreteParameter("Concentration", values=(0.057, 0.1, 0.153)),
        NumericalDiscreteParameter("Temp_C", values=(90, 105, 120)),
    ]

    objective = SingleTargetObjective(
        target=NumericalTarget(name="yield", mode=TargetMode.MAX)
    )

    campaign = Campaign(
        searchspace=SearchSpace.from_product(parameters=direct_arylation_mordred),
        objective=objective,
    )
    campaign_rand = Campaign(
        searchspace=SearchSpace.from_product(parameters=direct_arylation_ohe),
        recommender=RandomRecommender(),
        objective=objective,
    )
    lookup_direct_arylation = read_csv(
        PATH_PREFIX.joinpath("direct_arylation.csv").resolve()
    )
    batch_size = 3
    n_doe_iterations = 25
    n_mc_iterations = 50

    metadata = {
        "DOE_iterations": str(n_doe_iterations),
        "batch_size": str(batch_size),
        "n_mc_iterations": str(n_mc_iterations),
    }

    scenarios = {"Mordred Encoding": campaign, "Random Baseline": campaign_rand}
    return simulate_scenarios(
        scenarios,
        lookup_direct_arylation,
        batch_size=batch_size,
        n_doe_iterations=n_doe_iterations,
        n_mc_iterations=n_mc_iterations,
        impute_mode="error",
    ), metadata


benchmark_direct_arylation = SingleExecutionBenchmark(
    title="Direct Arylation Simulation maximum yield with Mordred",
    identifier=UUID("23df40f6-243c-49ca-ae71-81d733d8a88d"),
    benchmark_function=direct_arylation,
)
