"""
Test for imposing exclusion constraints on discrete parameters. For instance if
some parameter values are incompatible with certain values of another parameter
"""
import numpy as np
import pandas as pd

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise


def custom_function(ser: pd.Series) -> bool:
    """
    Example for a custom validator / filer
    """
    # Below we initialize the CUSTOM constraint with all the parameters this function
    # should have access to. The function can then compute a completely user-defined
    # validation of the searchspace points

    if ser.Solvent == "water":
        if ser.Temperature > 120 and ser.Concentration > 5:
            return False
        if ser.Temperature > 180 and ser.Concentration > 3:
            return False
    if ser.Solvent == "C3":
        if ser.Temperature < 150 and ser.Concentration > 3:
            return False
    return True


N_GRID_POINTS = 10
config_dict = {
    "project_name": "Custom Constraints Test (Discrete)",
    "allow_repeated_recommendations": False,
    "allow_recommending_already_measured": False,
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
            "name": "Concentration",
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
        # This constraint uses the user-defined function as a valdiator/filter
        {
            "type": "CUSTOM",
            "parameters": ["Concentration", "Solvent", "Temperature"],
            "validator": custom_function,
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
        "Number of entries with water, temp above 120 and concentration above 5:      ",
        (
            baybe_obj.searchspace.discrete.exp_rep["Concentration"].apply(
                lambda x: x > 5
            )
            & baybe_obj.searchspace.discrete.exp_rep["Temperature"].apply(
                lambda x: x > 120
            )
            & baybe_obj.searchspace.discrete.exp_rep["Solvent"].eq("water")
        ).sum(),
    )
    print(
        "Number of entries with water, temp above 180 and concentration above 3:      ",
        (
            baybe_obj.searchspace.discrete.exp_rep["Concentration"].apply(
                lambda x: x > 3
            )
            & baybe_obj.searchspace.discrete.exp_rep["Temperature"].apply(
                lambda x: x > 180
            )
            & baybe_obj.searchspace.discrete.exp_rep["Solvent"].eq("water")
        ).sum(),
    )
    print(
        "Number of entries with C3, temp above 180 and concentration above 3:         ",
        (
            baybe_obj.searchspace.discrete.exp_rep["Concentration"].apply(
                lambda x: x > 3
            )
            & baybe_obj.searchspace.discrete.exp_rep["Temperature"].apply(
                lambda x: x < 150
            )
            & baybe_obj.searchspace.discrete.exp_rep["Solvent"].eq("C3")
        ).sum(),
    )

    rec = baybe_obj.recommend(batch_quantity=5)

    add_fake_results(rec, baybe_obj)
    if kIter % 2:
        add_parameter_noise(rec, baybe_obj, noise_level=0.1)

    baybe_obj.add_results(rec)
