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
    "C4": "CCCC",
    "C5": "CCCCC",
    "c6": "c1ccccc1",
    "C6": "CCCCCC",
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
            # This constraint will only affect serchspace creation
            "type": "SUM_TARGET",
            "parameters": ["Fraction1", "Fraction2", "Fraction3"],
            "target_value": 100.0,
            # 'tolerance': 0.5,
        },
        {
            # This constraint will only affect serchspace creation
            "type": "MAX_N_DUPLICATES",
            "parameters": ["Solvent1", "Solvent2", "Solvent3"],
            "max_duplicates": 0,
        },
        {
            # This constraint will affect the modeling
            "type": "INVARIANCE",
            "parameters": ["Solvent1", "Solvent2", "Solvent3"],
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
    print(
        baybe_obj.searchspace_exp_rep.loc[
            ~baybe_obj.searchspace_metadata["dont_recommend"],
            ["Fraction1", "Fraction2", "Fraction3"],
        ].sum(axis=1)
    )
    print(
        baybe_obj.searchspace_exp_rep.loc[
            ~baybe_obj.searchspace_metadata["dont_recommend"],
            ["Solvent1", "Solvent2", "Solvent3"],
        ]
        .nunique(axis=1)
        .min()
    )

    rec = baybe_obj.recommend(batch_quantity=5)

    add_fake_results(rec, baybe_obj)
    if kIter % 2:
        add_parameter_noise(rec, baybe_obj, noise_level=0.1)

    baybe_obj.add_results(rec)
