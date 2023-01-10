"""
Test for imposing exclusion constraints on discrete parameters. For instance if
some parameter values are incompatible with certain values of another parameter
"""
import numpy as np

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise

N_GRID_POINTS = 15
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
            "values": list(np.linspace(100, 200, N_GRID_POINTS)),
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
    "constraints": [
        # This constraint simulates a situation where solvents C2 and C4 are not
        # compatible with temperatures > 154 and should thus be excluded
        {
            "type": "EXCLUDE",
            "parameters": ["Temperature", "Solvent"],
            "combiner": "AND",
            "conditions": [
                {
                    "type": "THRESHOLD",
                    "threshold": 151,
                    "operator": ">",
                },
                {
                    "type": "SUBSELECTION",
                    "selection": ["C2", "C4"],
                },
            ],
        },
        # This constraint simulates a situation where solvents C5 and C6 are not
        # compatible with pressures >= 5 and should thus be excluded
        {
            "type": "EXCLUDE",
            "parameters": ["Pressure", "Solvent"],
            "combiner": "AND",
            "conditions": [
                {
                    "type": "THRESHOLD",
                    "threshold": 5,
                    "operator": ">",
                },
                {
                    "type": "SUBSELECTION",
                    "selection": ["C5", "C6"],
                },
            ],
        },
        # This constraint simulates a situation where pressures below 3 should never be
        # combined with temperatures above 120
        {
            "type": "EXCLUDE",
            "parameters": ["Pressure", "Temperature"],
            "combiner": "AND",
            "conditions": [
                {
                    "type": "THRESHOLD",
                    "threshold": 3.0,
                    "operator": "<",
                },
                {
                    "type": "THRESHOLD",
                    "threshold": 120.0,
                    "operator": ">",
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

    print("### ASSERTS ###")
    print(
        "Number of entries with either Solvents C2 or C4 and a temperature above 151: ",
        (
            baybe_obj.searchspace.exp_rep["Temperature"].apply(lambda x: x > 151)
            & baybe_obj.searchspace.exp_rep["Solvent"].apply(
                lambda x: x in ["C2", "C4"]
            )
        ).sum(),
    )
    print(
        "Number of entries with either Solvents C5 or C6 and a pressure above 5:      ",
        (
            baybe_obj.searchspace.exp_rep["Pressure"].apply(lambda x: x > 5)
            & baybe_obj.searchspace.exp_rep["Solvent"].apply(
                lambda x: x in ["C5", "C6"]
            )
        ).sum(),
    )
    print(
        "Number of entries with pressure below 3 and temperature above 120:           ",
        (
            baybe_obj.searchspace.exp_rep["Pressure"].apply(lambda x: x < 3)
            & baybe_obj.searchspace.exp_rep["Temperature"].apply(lambda x: x > 120)
        ).sum(),
    )

    rec = baybe_obj.recommend(batch_quantity=5)

    add_fake_results(rec, baybe_obj)
    if kIter % 2:
        add_parameter_noise(rec, baybe_obj, noise_level=0.1)

    baybe_obj.add_results(rec)
