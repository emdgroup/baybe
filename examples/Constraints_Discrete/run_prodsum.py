"""
Test for imposing exclusion constraints on discrete parameters. For instance if
some parameter values are incompatible with certain values of another parameter
"""
import numpy as np

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise

N_GRID_POINTS = 5

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
            },
            "encoding": "RDKIT",
        },
        {
            "name": "SomeSetting",
            "type": "CAT",
            "values": ["slow", "normal", "fast"],
            "encoding": "INT",
        },
        {
            "name": "NumParameter1",
            "type": "NUM_DISCRETE",
            "values": list(np.linspace(0, 100, N_GRID_POINTS)),
            "tolerance": 0.5,
        },
        {
            "name": "NumParameter2",
            "type": "NUM_DISCRETE",
            "values": list(np.linspace(0, 100, N_GRID_POINTS)),
            "tolerance": 0.5,
        },
        {
            "name": "NumParameter3",
            "type": "NUM_DISCRETE",
            "values": list(np.linspace(0, 100, N_GRID_POINTS)),
            "tolerance": 0.5,
        },
        {
            "name": "NumParameter4",
            "type": "NUM_DISCRETE",
            "values": list(np.linspace(0, 100, N_GRID_POINTS)),
            "tolerance": 0.5,
        },
        {
            "name": "NumParameter5",
            "type": "NUM_DISCRETE",
            "values": list(np.linspace(0, 100, N_GRID_POINTS)),
            "tolerance": 0.5,
        },
        {
            "name": "NumParameter6",
            "type": "NUM_DISCRETE",
            "values": list(np.linspace(0, 100, N_GRID_POINTS)),
            "tolerance": 0.5,
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
    # The constrains test whether conditions relating to a sum and product are ensured
    "constraints": [
        {
            "type": "SUM",
            "parameters": ["NumParameter1", "NumParameter2"],
            "condition": {
                "threshold": 150.0,
                "operator": "<=",
            },
        },
        {
            "type": "PRODUCT",
            "parameters": ["NumParameter3", "NumParameter4"],
            "condition": {
                "threshold": 30.0,
                "operator": ">=",
            },
        },
        {
            "type": "SUM",
            "parameters": ["NumParameter5", "NumParameter6"],
            "condition": {
                "threshold": 100.0,
                "operator": "=",
                "tolerance": 1.0,
            },
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
        "Number of entries with 1,2-sum above 150:      ",
        (
            baybe_obj.searchspace.discrete.exp_rep[
                ["NumParameter1", "NumParameter2"]
            ].sum(axis=1)
            > 150.0
        ).sum(),
    )
    print(
        "Number of entries with 3,4-product under 30:   ",
        (
            baybe_obj.searchspace.discrete.exp_rep[
                ["NumParameter3", "NumParameter4"]
            ].prod(axis=1)
            < 30
        ).sum(),
    )
    print(
        "Number of entries with 5,6-sum unequal to 100: ",
        baybe_obj.searchspace.discrete.exp_rep[["NumParameter5", "NumParameter6"]]
        .sum(axis=1)
        .apply(lambda x: x - 100.0)
        .abs()
        .gt(0.01)
        .sum(),
    )

    rec = baybe_obj.recommend(batch_quantity=5)

    add_fake_results(rec, baybe_obj)
    if kIter % 2:
        add_parameter_noise(rec, baybe_obj, noise_level=0.1)

    baybe_obj.add_results(rec)
