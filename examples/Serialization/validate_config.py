"""
This example shows how to load and validate a user defined configuration file.
We use the two configuration dictionaries. The first one represents a valid
configuration, the second does not.
"""

from baybe import BayBE


CONFIG = """
{
    "parameters": [
        {
            "type": "CategoricalParameter",
            "name": "Granularity",
            "values": [
                "coarse",
                "fine",
                "ultra-fine"
            ],
            "encoding": "OHE"
        },
        {
            "type": "NumericalDiscreteParameter",
            "name": "Pressure[bar]",
            "values": [
                1,
                5,
                10
            ],
            "tolerance": 0.2
        },
        {
            "type": "SubstanceParameter",
            "name": "Solvent",
            "data": {
                "Solvent A": "COC",
                "Solvent B": "CCCCC",
                "Solvent C": "COCOC",
                "Solvent D": "CCOCCOCCN"
            },
            "decorrelate": true,
            "encoding": "MORDRED"
        }
    ],
    "constraints": [],
    "objective": {
        "mode": "SINGLE",
        "targets": [
            {
                "name": "Yield",
                "mode": "MAX"
            }
        ]
    },
    "strategy": {
        "initial_recommender": {
            "type": "FPSRecommender"
        },
        "recommender": {
            "type": "SequentialGreedyRecommender",
            "surrogate_model_cls": "GP",
            "acquisition_function_cls": "qEI"
        },
        "allow_repeated_recommendations": false,
        "allow_recommending_already_measured": false
    }
}
"""

INVALID_CONFIG = """
{
    "parameters": [
        {
            "type": "INVALID_TYPE",
            "name": "Granularity",
            "values": [
                "coarse",
                "fine",
                "ultra-fine"
            ],
            "encoding": "OHE"
        },
        {
            "type": "NumericalDiscreteParameter",
            "name": "Pressure[bar]",
            "values": [
                1,
                5,
                10
            ],
            "tolerance": 0.2
        },
        {
            "type": "SubstanceParameter",
            "name": "Solvent",
            "data": {
                "Solvent A": "COC",
                "Solvent B": "CCCCC",
                "Solvent C": "COCOC",
                "Solvent D": "CCOCCOCCN"
            },
            "decorrelate": true,
            "encoding": "MORDRED"
        }
    ],
    "constraints": [],
    "objective": {
        "mode": "SINGLE",
        "targets": [
            {
                "name": "Yield",
                "mode": "MAX"
            }
        ]
    },
    "strategy": {
        "initial_recommender": {
            "type": "FPSRecommender"
        },
        "recommender": {
            "type": "SequentialGreedyRecommender",
            "surrogate_model_cls": "GP",
            "acquisition_function_cls": "qEI"
        },
        "allow_repeated_recommendations": false,
        "allow_recommending_already_measured": false
    }
}
"""

# The first validation should work and print the line contained in the try block
try:
    BayBE.validate_config(CONFIG)
    print("Succesfully validated first config and created a BayBE object!")
except Exception:  # pylint: disable=W0702
    print("Something is wrong with config 1 which should not happen!")

# This should fail
try:
    BayBE.validate_config(INVALID_CONFIG)
    baybe = BayBE.from_config(INVALID_CONFIG)
except Exception:  # pylint: disable=W0702
    print("Something is wrong with config 2 which is what we epexted!")
