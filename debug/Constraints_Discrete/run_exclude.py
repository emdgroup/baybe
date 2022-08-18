"""
Test for imposing exclusion constraints on discrete parameters. For instance if
some parameter values are incompatible with certain values of another parameter
"""
import numpy as np

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise

# Simple example with one numerical target, two categorical and one numerical discrete
# parameter
config_dict = {
    "project_name": "Exclusion Constraints Test (Discrete)",
    "allow_repeated_recommendations": False,
    "allow_recommending_already_measured": True,
    "numerical_measurements_must_be_within_tolerance": True,
    "parameters": [
        {
            "name": "Solvent",
            "type": "SUBSTANCE",
            "data": {
                "water": "O",
                "C1": "C",
                "C2": "CC",
                "C3": "CCC",
                "C4": "CCCC",
                "C5": "CCCCC",
                "c6": "c1ccccc1",
                "C6": "CCCCCC",
            },
            "encoding": "RDKIT",
        },
        {
            "name": "SomeSetting",
            "type": "CAT",
            "values": ["very slow", "slow", "normal", "fast", "very fast"],
            "encoding": "INT",
        },
        {
            "name": "Temperature",
            "type": "NUM_DISCRETE",
            "values": list(np.linspace(100, 200, 21)),
            "tolerance": 0.5,
        },
        {
            "name": "Pressure",
            "type": "NUM_DISCRETE",
            "values": [1, 2, 5, 10],
            "tolerance": 0.4,
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
        # "surrogate_model_cls": "GP",
    },
    # The constrains simulate a situation where solvents C2 and C4 are not compatible
    # with temperatures > 154 and should thus be excluded
    "constraints": [
        {
            "type": "EXCLUDE",
            "combiner": "AND",
            "conditions": [
                {
                    "type": "THRESHOLD",
                    "parameter": "Temperature",
                    "threshold": 154,
                    "operator": ">",
                },
                {
                    "type": "SUBSELECTION",
                    "parameter": "Solvent",
                    "selection": ["C2", "C4"],
                },
            ],
        },
    ],
}

# Create BayBE object, add fake results and print what happens to internal data
config = BayBEConfig(**config_dict)
baybe_obj = BayBE(config)
print(baybe_obj)

N_ITERATIONS = 3
for kIter in range(N_ITERATIONS):
    print(f"\n\n##### ITERATION {kIter+1} #####")

    rec = baybe_obj.recommend(batch_quantity=5)

    add_fake_results(rec, baybe_obj)
    if kIter % 2:
        add_parameter_noise(rec, baybe_obj, noise_level=0.1)

    baybe_obj.add_results(rec)
