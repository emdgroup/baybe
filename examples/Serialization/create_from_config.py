## Example for creating campaigns from configs

# This example shows how to load a configuration file and use it to create a campaign.
# In such a configuration file, the objects used to create a campaign are represented by
# strings. We use the following configuration dictionaries, representing a valid campaign.

# Note that the json format is required for the config file.
# You can create such a config by providing a  dictionary with `"type":"name of the class"`.

### Necessary imports

from baybe import Campaign

### The configuration dictionary as a string

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

### Creating a campaign from the configuration file

# Although we know in this case that the config represents a valid configuration for a
# campaign. If the config is invalid an exception will be thrown.

campaign = Campaign.from_config(CONFIG)

# We now perform a recommendation as usual and print it.

recommendation = campaign.recommend(batch_size=3)
print(recommendation)
