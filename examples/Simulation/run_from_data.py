"""
Test for history simulation of a single target with a lookup from which parameter
values are inferred automatically. The user has to specify the parameter types.
Recommendations that have not been measured yet are ignored.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from baybe.simulation import simulate_from_data

lookup = pd.read_excel("./lookup_withmissing.xlsx")

config_dict_base = {
    "project_name": "Simulation from existing measurements",
    "allow_repeated_recommendations": False,
    "allow_recommending_already_measured": False,
    "numerical_measurements_must_be_within_tolerance": True,
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


results = simulate_from_data(
    config_base=config_dict_base,
    lookup=lookup,
    impute_mode="ignore",
    n_exp_iterations=15,
    n_mc_iterations=5,
    batch_quantity=2,
    parameter_types={
        "All Cat": [
            {"name": "Concentration", "type": "CAT"},
            {"name": "Temp_C", "type": "CAT"},
            {"name": "Solvent", "type": "CAT"},
        ],
        "Substance Cat": [
            {"name": "Concentration", "type": "NUM_DISCRETE"},
            {"name": "Temp_C", "type": "NUM_DISCRETE"},
            {"name": "Solvent", "type": "CAT"},
        ],
        "Substance Normal": [
            {"name": "Concentration", "type": "NUM_DISCRETE"},
            {"name": "Temp_C", "type": "NUM_DISCRETE"},
            {"name": "Solvent", "type": "SUBSTANCE", "encoding": "RDKIT"},
        ],
    },
)

print(results)

sns.lineplot(data=results, x="Num_Experiments", y="yield_CumBest", hue="Variant")
plt.gcf().set_size_inches(24, 8)
plt.savefig("./run_from_data.png")
