"""
Run history simulation for a Suzuki reaction where all possible combinations have
been measured
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from baybe.history import simulate_from_configs

lookup = pd.read_excel("./lookup.xlsx")

dict_solvent = {
    "DMF": r"O=CN(C)C",
    "MeCN": r"N#CC",
    "MeOH": r"CO",
    "THF": r"C1COCC1",
}

dict_base = {
    "CsF": r"[Cs+].[F-]",
    "Et3N": r"CCN(CC)CC",
    "K3PO4": r"O=P([O-])([O-])[O-].[K+].[K+].[K+]",
    "KOH": r"[K+].[OH-]",
    "LiOtBu": r"CC([O-])C.[Li+]",
    "NaHCO3": r"OC([O-])=O.[Na+]",
    "NaOH": r"[Na+].[OH-]",
}

dict_ligand = {
    "AmPhos": r"CC(C)(C)P(C(C)(C)C)C1=CC=C(N(C)C)C=C1",
    "CataCXiumA": r"CCCCP(C12C[C@@H]3C[C@@H](C[C@H](C2)C3)C1)C45C[C@H]6C[C@@H](C5)C"
    "[C@@H](C4)C6",
    "dppf": r"[c-]1(P(C2=CC=CC=C2)C3=CC=CC=C3)cccc1.[c-]4(P(C5=CC=CC=C5)C6=CC=CC=C6)"
    "cccc4.[Fe+2]",
    "dtbpf": r"CC(C)(P(C(C)(C)C)[c-]1cccc1)C.CC(C)(P(C(C)(C)C)[c-]2cccc2)C.[Fe+2]",
    "P(Cy)3": r"P(C1CCCCC1)(C2CCCCC2)C3CCCCC3",
    "P(o-Tol)3": r"CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C",
    "P(Ph)3": r"P(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3",
    "SPhos": r"COC1=CC=CC(OC)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2",
    "Xantphos": r"CC1(C)C2=C(OC3=C1C=CC=C3P(C4=CC=CC=C4)C5=CC=CC=C5)C(P(C6=CC=CC=C6)"
    "C7=CC=CC=C7)=CC=C2",
    "XPhos": r"CC(C1=C(C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C(C(C)C)=CC(C(C)C)=C1)C",
    "P(tBu)3": r"CC(P(C(C)(C)C)C(C)(C)C)(C)C",
}

dict_nucleophile = {
    "Ar2BF3": r"CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1[B-](F)(F)F",
    "Ar2BOH2": r"CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1B(O)O",
    "Ar2BPin": r"CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1B4OC(C)(C)C(C)(C)O4",
}

dict_electrophile = {
    "Ar1Br": r"BrC1=CC=C(N=CC=C2)C2=C1",
    "Ar1Cl": r"ClC1=CC=C(N=CC=C2)C2=C1",
    "Ar1I": r"IC1=CC=C(N=CC=C2)C2=C1",
    "Ar1OTf": r"O=S(OC1=CC=C(N=CC=C2)C2=C1)(C(F)(F)F)=O",
}

config_dict_base = {
    "project_name": "Aryl Amination",
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
            "name": "Nucleophile",
            "type": "SUBSTANCE",
            "data": dict_nucleophile,
            "encoding": "MORDRED",
        },
        {
            "name": "Electrophile",
            "type": "SUBSTANCE",
            "data": dict_electrophile,
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
            "name": "Solvent",
            "type": "SUBSTANCE",
            "data": dict_solvent,
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
        {
            "name": "Nucleophile",
            "type": "SUBSTANCE",
            "data": dict_nucleophile,
            "encoding": "RDKIT",
        },
        {
            "name": "Electrophile",
            "type": "SUBSTANCE",
            "data": dict_electrophile,
            "encoding": "RDKIT",
        },
    ],
}

config_dict_v3 = {
    "project_name": "GP | FP",
    "parameters": [
        {
            "name": "Solvent",
            "type": "SUBSTANCE",
            "data": dict_solvent,
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
        {
            "name": "Nucleophile",
            "type": "SUBSTANCE",
            "data": dict_nucleophile,
            "encoding": "MORGAN_FP",
        },
        {
            "name": "Electrophile",
            "type": "SUBSTANCE",
            "data": dict_electrophile,
            "encoding": "MORGAN_FP",
        },
    ],
}

config_dict_v4 = {
    "project_name": "GP | OHE",
    "parameters": [
        {
            "name": "Solvent",
            "type": "CAT",
            "values": list(dict_solvent.keys()),
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
        {
            "name": "Nucleophile",
            "type": "CAT",
            "values": list(dict_nucleophile.keys()),
            "encoding": "OHE",
        },
        {
            "name": "Electrophile",
            "type": "CAT",
            "values": list(dict_electrophile.keys()),
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
    lookup=None,
    n_exp_iterations=10,
    n_mc_iterations=5,
    batch_quantity=5,
    config_variants={
        "GP | Mordred": config_dict_v1,
        "GP | RDKit": config_dict_v2,
        "GP | FP": config_dict_v3,
        "GP | OHE": config_dict_v4,
        "RANDOM": config_dict_v5,
    },
)

print(results)

sns.lineplot(data=results, x="Num_Experiments", y="yield_CumBest", hue="Variant")
plt.gcf().set_size_inches(24, 8)
plt.savefig("./simulation.png")
