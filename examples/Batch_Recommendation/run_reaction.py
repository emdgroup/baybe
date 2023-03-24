"""
Compare different batch recommendation strategies on the direct arylation example
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from baybe.recommender import Recommender
from baybe.simulation import simulate_from_configs

lookup = pd.read_excel("../Reaction_DirectArylation/lookup.xlsx")

dict_solvent = {
    "DMAc": r"CC(N(C)C)=O",
    "Butyornitrile": r"CCCC#N",
    "Butyl Ester": r"CCCCOC(C)=O",
    "p-Xylene": r"CC1=CC=C(C)C=C1",
}

dict_base = {
    "Potassium acetate": r"O=C([O-])C.[K+]",
    "Potassium pivalate": r"O=C([O-])C(C)(C)C.[K+]",
    "Cesium acetate": r"O=C([O-])C.[Cs+]",
    "Cesium pivalate": r"O=C([O-])C(C)(C)C.[Cs+]",
}

dict_ligand = {
    "BrettPhos": r"CC(C)C1=CC(C(C)C)=C(C(C(C)C)=C1)C2=C(P(C3CCCCC3)C4CCCCC4)C(OC)="
    "CC=C2OC",
    "Di-tert-butylphenylphosphine": r"CC(C)(C)P(C1=CC=CC=C1)C(C)(C)C",
    "(t-Bu)PhCPhos": r"CN(C)C1=CC=CC(N(C)C)=C1C2=CC=CC=C2P(C(C)(C)C)C3=CC=CC=C3",
    "Tricyclohexylphosphine": r"P(C1CCCCC1)(C2CCCCC2)C3CCCCC3",
    "PPh3": r"P(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3",
    "XPhos": r"CC(C1=C(C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C(C(C)C)=CC(C(C)C)=C1)C",
    "P(2-furyl)3": r"P(C1=CC=CO1)(C2=CC=CO2)C3=CC=CO3",
    "Methyldiphenylphosphine": r"CP(C1=CC=CC=C1)C2=CC=CC=C2",
    "1268824-69-6": r"CC(OC1=C(P(C2CCCCC2)C3CCCCC3)C(OC(C)C)=CC=C1)C",
    "JackiePhos": r"FC(F)(F)C1=CC(P(C2=C(C3=C(C(C)C)C=C(C(C)C)C=C3C(C)C)C(OC)=CC=C2OC)"
    r"C4=CC(C(F)(F)F)=CC(C(F)(F)F)=C4)=CC(C(F)(F)F)=C1",
    "SCHEMBL15068049": r"C[C@]1(O2)O[C@](C[C@]2(C)P3C4=CC=CC=C4)(C)O[C@]3(C)C1",
    "Me2PPh": r"CP(C)C1=CC=CC=C1",
}

config_dict_base = {
    "project_name": "Direct Arylation",
    "allow_repeated_recommendations": False,
    "allow_recommending_already_measured": False,
    "numerical_measurements_must_be_within_tolerance": True,
    "parameters": [
        {
            "name": "Solvent",
            "type": "SUBSTANCE",
            "data": dict_solvent,
            "encoding": "MORDRED",
        },
        {
            "name": "Base",
            "type": "SUBSTANCE",
            "data": dict_base,
            "encoding": "MORDRED",
        },
        {
            "name": "Ligand",
            "type": "SUBSTANCE",
            "data": dict_ligand,
            "encoding": "MORDRED",
        },
        {
            "name": "Temp_C",
            "type": "NUM_DISCRETE",
            "values": [90, 105, 120],
            "tolerance": 2,
        },
        {
            "name": "Concentration",
            "type": "NUM_DISCRETE",
            "values": [0.057, 0.1, 0.153],
            "tolerance": 0.005,
        },
    ],
    "objective": {
        "mode": "SINGLE",
        "targets": [
            {
                "name": "yield",
                "type": "NUM",
                "mode": "MAX",
            },
        ],
    },
}


config_variants = {}
for recommender_type in Recommender.SUBCLASSES:
    if recommender_type == "SEQUENTIAL_GREEDY_CONTINUOUS":
        continue
    config_variants[recommender_type] = {
        "strategy": {
            "recommender_cls": recommender_type,
            "surrogate_model_cls": "GP",
            "acquisition_function_cls": "qEI",
        },
    }

results = simulate_from_configs(
    config_base=config_dict_base,
    lookup=lookup,
    impute_mode="worst",
    n_exp_iterations=12,
    n_mc_iterations=5,
    batch_quantity=5,
    config_variants=config_variants,
)

print(results)

max_yield = lookup["yield"].max()
sns.lineplot(
    data=results, x="Num_Experiments", y="yield_CumBest", hue="Variant", marker="x"
)
plt.plot([2, 5 * 10], [max_yield, max_yield], "--r")
plt.legend(loc="lower right")
plt.gcf().set_size_inches(20, 8)
plt.savefig("./run_reaction.png")
