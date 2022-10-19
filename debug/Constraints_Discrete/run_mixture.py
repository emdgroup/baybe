"""
Test for imposing sum constraints for discrete parameters, e.g. for mixture fractions
that need to sum up to 1
"""
import numpy as np

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise

solvs = {
    "water": "O",
    "C1": "C",
    "C2": "CC",
    "C3": "CCC",
}

N_GRID_POINTS = 5

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
            # This constraint will only affect searchspace creation
            "type": "SUM_TARGET",
            "parameters": ["Fraction1", "Fraction2", "Fraction3"],
            "target_value": 100.0,
            # 'tolerance': 0.5,
        },
        {
            # This constraint will only affect serchspace creation
            "type": "NO_LABEL_DUPLICATES",
            "parameters": ["Solvent1", "Solvent2", "Solvent3"],
        },
        {
            # This constraint will affect searchspace creation
            "type": "PERMUTATION_INVARIANCE",
            "parameters": ["Solvent1", "Solvent2", "Solvent3"],
        },
    ],
}

# Create BayBE object, add fake results and print what happens to internal data
config = BayBEConfig(**config_dict)
baybe_obj = BayBE(config)
print(baybe_obj)

N_ITERATIONS = 3
print("\n\n######## ALL FOLLOWING OUTPUTS SHOULD BE 0 ########")
for kIter in range(N_ITERATIONS):
    print(f"\n##### ITERATION {kIter+1} #####")
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

    rec = baybe_obj.recommend(batch_quantity=5)

    add_fake_results(rec, baybe_obj)
    if kIter % 2:
        add_parameter_noise(rec, baybe_obj, noise_level=0.1)

    baybe_obj.add_results(rec)
