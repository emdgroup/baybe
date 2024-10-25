## Example for validation of a config file

# This example shows how to load and validate a user defined configuration file.
# We use the two configuration dictionaries.
# The first one represents a valid configuration, the second does not.

### Necessary imports

from cattrs import ClassValidationError

from baybe import Campaign

### Defining config dictionaries

# Note that the following explicit call `str()` is not strictly necessary.
# It is included since our method of converting this example to a markdown file does not
# interpret this part of the code as `python` code if we do not include this call.

CONFIG = str(
    """
{
    "searchspace": {
        "constructor": "from_product",
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
        "constraints": []
    },
    "objective": {
        "type": "SingleTargetObjective",
        "target":
            {
                "type": "NumericalTarget",
                "name": "Yield",
                "mode": "MAX"
            }
    },
    "recommender": {
        "type": "TwoPhaseMetaRecommender",
        "initial_recommender": {
            "type": "FPSRecommender"
        },
        "recommender": {
            "type": "BotorchRecommender",
            "surrogate_model": {
                "type": "GaussianProcessSurrogate"
            },
            "acquisition_function": "qEI",
            "allow_repeated_recommendations": false,
            "allow_recommending_already_measured": false
        },
        "switch_after": 1
    }
}
"""
)

INVALID_CONFIG = str(
    """
{
    "searchspace": {
        "constructor": "from_product",
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
        "constraints": []
    },
    "objective": {
        "type": "SingleTargetObjective",
        "target":
            {
                "type": "NumericalTarget",
                "name": "Yield",
                "mode": "MAX"
            }
    },
    "recommender": {
        "type": "TwoPhaseMetaRecommender",
        "initial_recommender": {
            "type": "FPSRecommender"
        },
        "recommender": {
            "type": "BotorchRecommender",
            "surrogate_model": {
                "type": "GaussianProcessSurrogate"
            },
            "acquisition_function": "qEI",
            "allow_repeated_recommendations": false,
            "allow_recommending_already_measured": false
        }
    }
}
"""
)

### Verification of the two dictionaries

# The first validation should work.

Campaign.validate_config(CONFIG)
print("The first config seems valid.")

# This should fail.

try:
    Campaign.validate_config(INVALID_CONFIG)
    campaign = Campaign.from_config(INVALID_CONFIG)
except ClassValidationError:
    print("Something is wrong with the second config, which is what we expected!")
