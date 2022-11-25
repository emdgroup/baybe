"""
Test for history simulation of a single target without a lookup, ie random addition
of measurements.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from baybe.simulation import simulate_from_configs

substances = {
    "Water": "O",
    "THF": "C1CCOC1",
    "DMF": "CN(C)C=O",
    "Hexane": "CCCCCC",
    "Ethanol": "CCO",
}

config_dict_base = {
    "project_name": "Clustering Test",
    "allow_repeated_recommendations": False,
    "allow_recommending_already_measured": False,
    "numerical_measurements_must_be_within_tolerance": True,
    "parameters": [
        {
            "name": "Flow_Strength",
            "type": "CAT",
            "values": ["very low", "low", "medium", "strong", "very strong"],
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
        "mode": "SINGLE",
        "targets": [
            {
                "name": "Target_1",
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
    "project_name": "PAM",
    "strategy": {
        "surrogate_model_cls": "GP",
        "recommender_cls": "UNRESTRICTED_RANKING",
        "initial_strategy": "PAM",
    },
}

config_dict_v2 = {
    "project_name": "K-Means",
    "strategy": {
        "surrogate_model_cls": "GP",
        "recommender_cls": "UNRESTRICTED_RANKING",
        "initial_strategy": "KMEANS",
    },
}

config_dict_v3 = {
    "project_name": "Gaussian Mixture",
    "strategy": {
        "surrogate_model_cls": "GP",
        "recommender_cls": "UNRESTRICTED_RANKING",
        "initial_strategy": "GMM",
    },
}

config_dict_v4 = {
    "project_name": "Random",
    "strategy": {
        "surrogate_model_cls": "GP",
        "recommender_cls": "UNRESTRICTED_RANKING",
        "initial_strategy": "RANDOM",
    },
}


results = simulate_from_configs(
    config_base=config_dict_base,
    lookup=None,
    n_exp_iterations=15,
    n_mc_iterations=200,
    batch_quantity=5,
    config_variants={
        "PAM": config_dict_v1,
        "KMEANS": config_dict_v2,
        "GMM": config_dict_v3,
        "RANDOM": config_dict_v4,
    },
)

print(results)

sns.lineplot(data=results, x="Num_Experiments", y="Target_1_CumBest", hue="Variant")
plt.gcf().set_size_inches(22, 8)
plt.savefig("./run_simulation.png")
