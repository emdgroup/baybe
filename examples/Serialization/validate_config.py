### Example for validation of a config file

"""
This example shows how to load and validate a user defined configuration file.
We use the two configuration dictionaries.
The first one represents a valid configuration, the second does not.
"""

#### Necessary imports

from baybe import BayBE
from cattrs import ClassValidationError

#### Defining config dictionaries

# Note that the following explicit call `str()` is not strictly necessary.
# It is included since our method of converting this example to a markdown file does not interpret
# this part of the code as `python` code if we do not include this call.

CONFIG = str(
    """
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
)

INVALID_CONFIG = str(
    """
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
)

#### Verifictation of the two dictionaries

# The first validation should work
BayBE.validate_config(CONFIG)
print("Successfully validated first config and created a BayBE object!")

# This should fail.
try:
    BayBE.validate_config(INVALID_CONFIG)
    baybe = BayBE.from_config(INVALID_CONFIG)
except ClassValidationError:
    print("Something is wrong with config 2 which is what we expected!")
