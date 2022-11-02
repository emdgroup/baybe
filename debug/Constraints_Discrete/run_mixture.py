"""
Test for imposing sum constraints for discrete parameters, e.g. for mixture fractions
that need to sum up to 1
"""
import math

import numpy as np

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise

solvs = {
    "water": "O",
    "C1": "C",
    "C2": "CC",
    "C3": "CCC",
    # "C4": "CCCC",
    # "C5": "CCCCC",
}

N_GRID_POINTS = 7

# Simple example with one numerical target, two categorical and one numerical discrete
# parameter
config_dict = {
    "project_name": "Exclusion Constraints Test (Discrete)",
    "allow_repeated_recommendations": False,
    "allow_recommending_already_measured": True,
    "numerical_measurements_must_be_within_tolerance": True,
    "parameters": [
        {
            "name": "Solvent1",
            "type": "SUBSTANCE",
            "data": solvs,
            "encoding": "MORDRED",
        },
        {
            "name": "Solvent2",
            "type": "SUBSTANCE",
            "data": solvs,
            "encoding": "MORDRED",
        },
        {
            "name": "Solvent3",
            "type": "SUBSTANCE",
            "data": solvs,
            "encoding": "MORDRED",
        },
        {
            "name": "Fraction1",
            "type": "NUM_DISCRETE",
            "values": list(np.linspace(0, 100, N_GRID_POINTS)),
            "tolerance": 0.2,
        },
        {
            "name": "Fraction2",
            "type": "NUM_DISCRETE",
            "values": list(np.linspace(0, 100, N_GRID_POINTS)),
            "tolerance": 0.2,
        },
        {
            "name": "Fraction3",
            "type": "NUM_DISCRETE",
            "values": list(np.linspace(0, 100, N_GRID_POINTS)),
            "tolerance": 0.2,
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
    # The constrains simulate a situation where we want to mix up to three solvents,
    # but their respective fractions need to sum up to 100. Also, the solvents should
    # never be chosen twice.
    "constraints": [
        {
            # This constraint will affect searchspace creation
            "type": "PERMUTATION_INVARIANCE",
            "parameters": ["Solvent1", "Solvent2", "Solvent3"],
        },
        {
            # This constraint will only affect searchspace creation
            "type": "SUM",
            "parameters": ["Fraction1", "Fraction2", "Fraction3"],
            "condition": {
                "threshold": 100.0,
                "operator": "=",
                "tolerance": 1.0,
            },
        },
        {
            # This constraint will only affect searchspace creation
            "type": "NO_LABEL_DUPLICATES",
            "parameters": ["Solvent1", "Solvent2", "Solvent3"],
        },
        {
            # Test specifying dependencies. Also test the possibility to specify via
            # different conditions
            "type": "DEPENDENCIES",
            "parameters": ["Fraction1", "Fraction2", "Fraction3"],
            "conditions": [
                {
                    "type": "THRESHOLD",
                    "threshold": 0.0,
                    "operator": ">",
                },
                {
                    "type": "THRESHOLD",
                    "threshold": 0.0,
                    "operator": ">",
                },
                {
                    "type": "SUBSELECTION",
                    "selection": list(np.linspace(0, 100, N_GRID_POINTS)[1:]),
                },
            ],
            "affected_parameters": [
                ["Solvent1"],
                ["Solvent2"],
                ["Solvent3"],
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
    print(f"\n##### ITERATION {kIter+1} #####")

    print("### ASSERTS ###")
    print(
        "Number of searchspace entries where fractions do not sum to 100.0:      ",
        baybe_obj.searchspace_exp_rep[["Fraction1", "Fraction2", "Fraction3"]]
        .sum(axis=1)
        .ne(100.0)
        .sum(),
    )
    print(
        "Number of searchspace entries that have duplicate solvent labels:       ",
        baybe_obj.searchspace_exp_rep[["Solvent1", "Solvent2", "Solvent3"]]
        .nunique(axis=1)
        .ne(3)
        .sum(),
    )
    print(
        "Number of searchspace entries with permutation-invariant combinations:  ",
        baybe_obj.searchspace_exp_rep[["Solvent1", "Solvent2", "Solvent3"]]
        .apply(frozenset, axis=1)
        .to_frame()
        .join(baybe_obj.searchspace_exp_rep[["Fraction1", "Fraction2", "Fraction3"]])
        .duplicated()
        .sum(),
    )
    # The following asserts only work if N_GRID_POINTS splits the range into partitions
    # that can be displayed without rounding errors. If this is not the case
    # (for instance with N_GRID_POINTS = 13) then the sum constraint will throw out
    # entries that do not exactly match. This can be solved by using a tolerance or
    # better values for N_GRID_POINTS. The reference values are the number of
    # X-combinations times the number of possible partitions for X entries
    # (depends on N_GRID_POINTS)
    print(
        f"Number of unique 1-solvent entries (expected {math.comb(len(solvs), 1)*1})",
        (baybe_obj.searchspace_exp_rep[["Fraction1", "Fraction2", "Fraction3"]] == 0.0)
        .sum(axis=1)
        .eq(2)
        .sum(),
    )
    print(
        f"Number of unique 2-solvent entries (expected"
        f" {math.comb(len(solvs), 2)*(N_GRID_POINTS-2)})",
        (baybe_obj.searchspace_exp_rep[["Fraction1", "Fraction2", "Fraction3"]] == 0.0)
        .sum(axis=1)
        .eq(1)
        .sum(),
    )
    print(
        f"Number of unique 3-solvent entries (expected"
        f" {math.comb(len(solvs), 3)*((N_GRID_POINTS-3)*(N_GRID_POINTS-2))//2})",
        (baybe_obj.searchspace_exp_rep[["Fraction1", "Fraction2", "Fraction3"]] == 0.0)
        .sum(axis=1)
        .eq(0)
        .sum(),
    )

    rec = baybe_obj.recommend(batch_quantity=5)

    add_fake_results(rec, baybe_obj)
    if kIter % 2:
        add_parameter_noise(rec, baybe_obj, noise_level=0.1)

    baybe_obj.add_results(rec)
