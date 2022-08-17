"""
Test for history simulation of a single target without a lookup, ie random addition
of measurements.
"""

import logging

import matplotlib.pyplot as plt
import seaborn as sns
from baybe.simulation import simulate_from_configs

log = logging.getLogger()
log.setLevel(logging.CRITICAL)

substances = {
    "Water": "O",
    "THF": "C1CCOC1",
    "DMF": "CN(C)C=O",
    "Hexane": "CCCCCC",
    "Ethanol": "CCO",
}

config_dict_base = {
    "project_name": "Simulation Test",
    "allow_repeated_recommendations": False,
    "allow_recommending_already_measured": False,
    "numerical_measurements_must_be_within_tolerance": True,
    "parameters": [
        {
            "name": "Flow_Strength",
            "type": "CAT",
            "values": ["very bad", "bad", "OK", "good", "very good"],
            "encoding": "INT",
        },
        {
            "name": "Temperature",
            "type": "NUM_DISCRETE",
            "values": [100, 110, 120, 130, 140, 150, 160],
            "tolerance": 3,
        },
        {
            "name": "Substance_1",
            "type": "SUBSTANCE",
            "data": substances,
            "encoding": "MORDRED",
        },
    ],
    "objective": {
        "mode": "DESIRABILITY",
        "targets": [
            {
                "name": "TargetMax",
                "type": "NUM",
                "mode": "MAX",
                "bounds": (0, 100),
            },
            {
                "name": "TargetMin",
                "type": "NUM",
                "mode": "MIN",
                "bounds": (0, 100),
            },
            {
                "name": "TargetMatch",
                "type": "NUM",
                "mode": "MATCH",
                "bounds": (40, 60),
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
            "name": "Flow_Strength",
            "type": "CAT",
            "values": ["very bad", "bad", "OK", "good", "very good"],
            "encoding": "INT",
        },
        {
            "name": "Temperature",
            "type": "NUM_DISCRETE",
            "values": [100, 110, 120, 130, 140, 150, 160],
            "tolerance": 3,
        },
        {
            "name": "Substance_1",
            "type": "SUBSTANCE",
            "data": substances,
            "encoding": "RDKIT",
        },
    ],
}

config_dict_v3 = {
    "project_name": "GP | FP",
    "parameters": [
        {
            "name": "Flow_Strength",
            "type": "CAT",
            "values": ["very bad", "bad", "OK", "good", "very good"],
            "encoding": "INT",
        },
        {
            "name": "Temperature",
            "type": "NUM_DISCRETE",
            "values": [100, 110, 120, 130, 140, 150, 160],
            "tolerance": 3,
        },
        {
            "name": "Substance_1",
            "type": "SUBSTANCE",
            "data": substances,
            "encoding": "MORGAN_FP",
        },
    ],
}

config_dict_v4 = {
    "project_name": "GP | OHE",
    "parameters": [
        {
            "name": "Flow_Strength",
            "type": "CAT",
            "values": ["very bad", "bad", "OK", "good", "very good"],
            "encoding": "INT",
        },
        {
            "name": "Temperature",
            "type": "NUM_DISCRETE",
            "values": [100, 110, 120, 130, 140, 150, 160],
            "tolerance": 3,
        },
        {
            "name": "Substance_1",
            "type": "CAT",
            "values": list(substances.keys()),
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
    n_exp_iterations=15,
    n_mc_iterations=50,
    batch_quantity=3,
    config_variants={
        "GP | Mordred": config_dict_v1,
        "GP | RDKit": config_dict_v2,
        "GP | FP": config_dict_v3,
        "GP | OHE": config_dict_v4,
        "RANDOM": config_dict_v5,
    },
)

print(results)

plt.subplot(3, 1, 1)
sns.lineplot(
    data=results,
    x="Num_Experiments",
    y="TargetMax_CumBest",
    hue="Variant",
    markers=True,
)

plt.subplot(3, 1, 2)
sns.lineplot(
    data=results,
    x="Num_Experiments",
    y="TargetMin_CumBest",
    hue="Variant",
    markers=True,
)

plt.subplot(3, 1, 3)
sns.lineplot(
    data=results,
    x="Num_Experiments",
    y="TargetMatch_CumBest",
    hue="Variant",
    markers=True,
)

plt.gcf().set_size_inches(24, 24)
plt.savefig("./run_desirability.png")
