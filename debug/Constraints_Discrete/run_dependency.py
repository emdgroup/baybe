"""
Test for imposing dependency constraints
"""
from copy import deepcopy

import numpy as np

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise

solvs = {
    "water": "O",
    "C1": "C",
}

N_GRID_POINTS = 7

config_dict = {
    "project_name": "Exclusion Constraints Test (Discrete)",
    "allow_repeated_recommendations": False,
    "allow_recommending_already_measured": True,
    "numerical_measurements_must_be_within_tolerance": True,
    "parameters": [
        {
            "name": "Switch1",
            "type": "CAT",
            "values": ["on", "off"],
        },
        {
            "name": "Switch2",
            "type": "CAT",
            "values": ["left", "right"],
        },
        {
            "name": "Fraction1",
            "type": "NUM_DISCRETE",
            "values": list(np.linspace(0, 100, N_GRID_POINTS)),
            "tolerance": 0.2,
        },
        {
            "name": "Solvent1",
            "type": "SUBSTANCE",
            "data": solvs,
            "encoding": "MORDRED",
        },
        {
            "name": "FrameA",
            "type": "CAT",
            "values": ["A", "B"],
        },
        {
            "name": "FrameB",
            "type": "CAT",
            "values": ["A", "B"],
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
}

constraints1 = [
    {
        "type": "DEPENDENCIES",
        "parameters": ["Switch1", "Switch2"],
        "conditions": [
            {
                "type": "SUBSELECTION",
                "selection": ["on"],
            },
            {
                "type": "SUBSELECTION",
                "selection": ["right"],
            },
        ],
        "affected_parameters": [
            ["Solvent1", "Fraction1"],
            ["FrameA", "FrameB"],
        ],
    },
]

constraints2 = [
    {
        "type": "DEPENDENCIES",
        "parameters": ["Switch1"],
        "conditions": [
            {
                "type": "SUBSELECTION",
                "selection": ["on"],
            },
        ],
        "affected_parameters": [
            ["Solvent1", "Fraction1"],
        ],
    },
    {
        "type": "DEPENDENCIES",
        "parameters": ["Switch2"],
        "conditions": [
            {
                "type": "SUBSELECTION",
                "selection": ["right"],
            },
        ],
        "affected_parameters": [
            ["FrameA", "FrameB"],
        ],
    },
]

N_ITERATIONS = 2

# Create BayBE object, add fake results and print what happens to internal data
conf1 = deepcopy(config_dict)
conf1["constraints"] = constraints1
config = BayBEConfig(**conf1)
baybe_obj = BayBE(config)
for kIter in range(N_ITERATIONS):
    print(f"\n##### Version1 ITERATION {kIter+1} #####")

    print("### ASSERTS ###")
    print(
        f"Number entries with both switches on "
        f"(expected {N_GRID_POINTS*len(solvs)*2*2}): ",
        (
            (baybe_obj.searchspace_exp_rep["Switch1"] == "on")
            & (baybe_obj.searchspace_exp_rep["Switch2"] == "right")
        ).sum(),
    )
    print(
        f"Number entries with Switch1 off " f"(expected {2*2}):       ",
        (
            (baybe_obj.searchspace_exp_rep["Switch1"] == "off")
            & (baybe_obj.searchspace_exp_rep["Switch2"] == "right")
        ).sum(),
    )
    print(
        f"Number entries with Switch2 off "
        f"(expected {N_GRID_POINTS*len(solvs)}):"
        f"      ",
        (
            (baybe_obj.searchspace_exp_rep["Switch1"] == "on")
            & (baybe_obj.searchspace_exp_rep["Switch2"] == "left")
        ).sum(),
    )
    print(
        "Number entries with both switches off (expected 1): ",
        (
            (baybe_obj.searchspace_exp_rep["Switch1"] == "off")
            & (baybe_obj.searchspace_exp_rep["Switch2"] == "left")
        ).sum(),
    )

    rec = baybe_obj.recommend(batch_quantity=5)

    add_fake_results(rec, baybe_obj)
    if kIter % 2:
        add_parameter_noise(rec, baybe_obj, noise_level=0.1)

    baybe_obj.add_results(rec)


conf2 = deepcopy(config_dict)
conf2["constraints"] = constraints2
config2 = BayBEConfig(**conf2)
baybe_obj2 = BayBE(config2)
for kIter in range(N_ITERATIONS):
    print(f"\n##### Version2 ITERATION {kIter+1} #####")

    print("### ASSERTS ###")
    print(
        f"Number entries with both switches on "
        f"(expected {N_GRID_POINTS*len(solvs)*2*2}): ",
        (
            (baybe_obj2.searchspace_exp_rep["Switch1"] == "on")
            & (baybe_obj2.searchspace_exp_rep["Switch2"] == "right")
        ).sum(),
    )
    print(
        f"Number entries with Switch1 off " f"(expected {2*2}):       ",
        (
            (baybe_obj2.searchspace_exp_rep["Switch1"] == "off")
            & (baybe_obj2.searchspace_exp_rep["Switch2"] == "right")
        ).sum(),
    )
    print(
        f"Number entries with Switch2 off "
        f"(expected {N_GRID_POINTS*len(solvs)}):"
        f"      ",
        (
            (baybe_obj2.searchspace_exp_rep["Switch1"] == "on")
            & (baybe_obj2.searchspace_exp_rep["Switch2"] == "left")
        ).sum(),
    )
    print(
        "Number entries with both switches off (expected 1): ",
        (
            (baybe_obj2.searchspace_exp_rep["Switch1"] == "off")
            & (baybe_obj2.searchspace_exp_rep["Switch2"] == "left")
        ).sum(),
    )

    rec = baybe_obj2.recommend(batch_quantity=5)

    add_fake_results(rec, baybe_obj2)
    if kIter % 2:
        add_parameter_noise(rec, baybe_obj2, noise_level=0.1)

    baybe_obj2.add_results(rec)
