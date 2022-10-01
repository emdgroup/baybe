"""
Run history simulation for an aryl amination where all possible combinations have been
measured
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from baybe.simulation import simulate_from_configs

lookup = pd.read_excel("./lookup.xlsx")

dict_additives = {
    "3,5-dimethylisoxazole": r"Cc1onc(C)c1",
    "3-methyl-5-phenylisoxazole": r"Cc1cc(on1)c2ccccc2",
    "3-methylisoxazole": r"Cc1ccon1",
    "3-phenylisoxazole": r"o1ccc(n1)c2ccccc2",
    "4-phenylisoxazole": r"o1cc(cn1)c2ccccc2",
    "5-(2,6-difluorophenyl)isoxazole": r"Fc1cccc(F)c1c2oncc2",
    "5-methyl-3-(1H-pyrrol-1-yl)isoxazole": r"Cc1onc(c1)n2cccc2",
    "5-methylisoxazole": r"Cc1oncc1",
    "5-phenylisoxazole": r"o1nccc1c2ccccc2",
    "benzo[c]isoxazole": r"o1cc2ccccc2n1",
    "benzo[d]isoxazole": r"o1ncc2ccccc12",
    "ethyl-3-methoxyisoxazole-5-carboxylate": r"CCOC(=O)c1onc(OC)c1",
    "ethyl-3-methylisoxazole-5-carboxylate": r"CCOC(=O)c1onc(C)c1",
    "ethyl-5-methylisoxazole-3-carboxylate": r"CCOC(=O)c1cc(C)on1",
    "ethyl-5-methylisoxazole-4-carboxylate": r"CCOC(=O)c1cnoc1C",
    "ethyl-isoxazole-3-carboxylate": r"CCOC(=O)c1ccon1",
    "ethyl-isoxazole-4-carboxylate": r"CCOC(=O)c1conc1",
    "methyl-5-(furan-2-yl)isoxazole-3-carboxylate": r"COC(=O)c1cc(on1)c2occc2",
    "methyl-5-(thiophen-2-yl)isoxazole-3-carboxylate": r"COC(=O)c1cc(on1)c2sccc2",
    "methyl-isoxazole-5-carboxylate": r"COC(=O)c1oncc1",
    "N,N-dibenzylisoxazol-3-amine": r"C(N(Cc1ccccc1)c2ccon2)c3ccccc3",
    "N,N-dibenzylisoxazol-5-amine": r"C(N(Cc1ccccc1)c2oncc2)c3ccccc3",
}

dict_aryl_halide = {
    "1-bromo-4-ethylbenzene": r"CCc1ccc(Br)cc1",
    "1-bromo-4-methoxybenzene": r"COc1ccc(Br)cc1",
    "1-bromo-4-(trifluoromethyl)benzene": r"FC(F)(F)c1ccc(Br)cc1",
    "1-chloro-4-ethylbenzene": r"CCc1ccc(Cl)cc1",
    "1-chloro-4-methoxybenzene": r"COc1ccc(Cl)cc1",
    "1-chloro-4-(trifluoromethyl)benzene": r"FC(F)(F)c1ccc(Cl)cc1",
    "1-ethyl-4-iodobenzene": r"CCc1ccc(I)cc1",
    "1-iodo-4-methoxybenzene": r"Oc1ccc(I)cc1",
    "1-iodo-4-(trifluoromethyl)benzene": r"FC(F)(F)c1ccc(I)cc1",
    "2-bromopyridine": r"Brc1ccccn1",
    "2-chloropyridine": r"Clc1ccccn1",
    "2-iodopyridine": r"Ic1ccccn1",
    "3-bromopyridine": r"Brc1cccnc1",
    "3-chloropyridine": r"Clc1cccnc1",
    "3-iodopyridine": r"Ic1cccnc1",
}

dict_base = {
    "BTMG": r"CN(C)/C(N(C)C)=N\C(C)(C)C",
    "MTBD": r"CN1CCCN2CCCN=C12",
    "P2Et": r"CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC",
}

dict_ligand = {
    "Pd0-Ad-BrettPhos": r"CC(C1=C(C2=C(OC)C=CC(OC)=C2P(C34CC5CC(C4)CC(C5)C3)C67CC8CC"
    r"(C7)CC(C8)C6)C(C(C)C)=CC(C(C)C)=C1)C",
    "Pd0-t-Bu-BrettPhos": r"CC(C)C1=CC(C(C)C)=CC(C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C"
    r"(OC)=CC=C2OC",
    "Pd0-t-Bu-X-Phos": r"CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=CC=CC=C2P(C(C)(C)C)C(C)(C)C",
    "Pd0-X-Phos": r"CC(C)C1=CC(C(C)C)=CC(C(C)C)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2",
}

config_dict_base = {
    "project_name": "Aryl Amination",
    "allow_repeated_recommendations": False,
    "allow_recommending_already_measured": False,
    "numerical_measurements_must_be_within_tolerance": True,
    "parameters": [
        {
            "name": "Additive",
            "type": "SUBSTANCE",
            "data": dict_additives,
            "encoding": "MORDRED",
        },
        {
            "name": "Aryl_halide",
            "type": "SUBSTANCE",
            "data": dict_aryl_halide,
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
    "strategy": {
        "surrogate_model_cls": "GP",
        "recommender_cls": "UNRESTRICTED_RANKING",
    },
}

config_dict_v1 = {
    "project_name": "GP | Mordred",
}

config_dict_v2 = {
    "project_name": "GP | RDKit",
    "parameters": [
        {
            "name": "Additive",
            "type": "SUBSTANCE",
            "data": dict_additives,
            "encoding": "RDKIT",
        },
        {
            "name": "Aryl_halide",
            "type": "SUBSTANCE",
            "data": dict_aryl_halide,
            "encoding": "RDKIT",
        },
        {
            "name": "Base",
            "type": "SUBSTANCE",
            "data": dict_base,
            "encoding": "RDKIT",
        },
        {
            "name": "Ligand",
            "type": "SUBSTANCE",
            "data": dict_ligand,
            "encoding": "RDKIT",
        },
    ],
}

config_dict_v3 = {
    "project_name": "GP | FP",
    "parameters": [
        {
            "name": "Additive",
            "type": "SUBSTANCE",
            "data": dict_additives,
            "encoding": "MORGAN_FP",
        },
        {
            "name": "Aryl_halide",
            "type": "SUBSTANCE",
            "data": dict_aryl_halide,
            "encoding": "MORGAN_FP",
        },
        {
            "name": "Base",
            "type": "SUBSTANCE",
            "data": dict_base,
            "encoding": "MORGAN_FP",
        },
        {
            "name": "Ligand",
            "type": "SUBSTANCE",
            "data": dict_ligand,
            "encoding": "MORGAN_FP",
        },
    ],
}

config_dict_v4 = {
    "project_name": "GP | OHE",
    "parameters": [
        {
            "name": "Additive",
            "type": "CAT",
            "values": list(dict_additives.keys()),
            "encoding": "OHE",
        },
        {
            "name": "Aryl_halide",
            "type": "CAT",
            "values": list(dict_aryl_halide.keys()),
            "encoding": "OHE",
        },
        {
            "name": "Base",
            "type": "CAT",
            "values": list(dict_base.keys()),
            "encoding": "OHE",
        },
        {
            "name": "Ligand",
            "type": "CAT",
            "values": list(dict_ligand.keys()),
            "encoding": "OHE",
        },
    ],
}

config_dict_v5 = {
    "project_name": "Random",
    "strategy": {
        "recommender_cls": "RANDOM",
    },
}


results = simulate_from_configs(
    config_base=config_dict_base,
    lookup=lookup,
    impute_mode="mean",
    n_exp_iterations=30,
    n_mc_iterations=200,
    batch_quantity=2,
    config_variants={
        "GP | Mordred": config_dict_v1,
        "GP | RDKit": config_dict_v2,
        "GP | FP": config_dict_v3,
        "GP | OHE": config_dict_v4,
        "RANDOM": config_dict_v5,
    },
)

print(results)

max_yield = lookup["yield"].max()
sns.lineplot(
    data=results, x="Num_Experiments", y="yield_CumBest", hue="Variant", marker="x"
)
plt.plot([2, 2 * 30], [max_yield, max_yield], "--r")
plt.legend(loc="lower right")
plt.gcf().set_size_inches(20, 8)
plt.savefig("./simulation_encodings.png")
