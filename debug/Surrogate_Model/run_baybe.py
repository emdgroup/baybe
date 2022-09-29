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
    "project_name": "GP",
    "strategy": {
        "surrogate_model_cls": "GP",
        "recommender_cls": "UNRESTRICTED_RANKING",
    }
}

config_dict_v2 = {
    "project_name": "RF",
    "strategy": {
        "surrogate_model_cls": "RF",
        "recommender_cls": "UNRESTRICTED_RANKING",
    }
}

config_dict_v3 = {
    "project_name": "NG",
    "strategy": {
        "surrogate_model_cls": "NG",
        "recommender_cls": "UNRESTRICTED_RANKING",
    }
}

config_dict_v4 = {
    "project_name": "BL",
    "strategy": {
        "surrogate_model_cls": "BL",
        "recommender_cls": "UNRESTRICTED_RANKING",
    }
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
    n_exp_iterations=25,
    n_mc_iterations=5,
    batch_quantity=1,
    config_variants={
        "GP": config_dict_v1,
        "RF": config_dict_v2,
        "NG": config_dict_v3,
        "BL": config_dict_v4,
        "RANDOM": config_dict_v5,
    },
)

print(results)
results.to_csv("model_results.txt")

sns.lineplot(data=results, x="Num_Experiments", y="Target_1_CumBest", hue="Variant")
plt.gcf().set_size_inches(24, 8)
plt.savefig("./run_models.png")
