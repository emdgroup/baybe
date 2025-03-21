## Example for custom parameter passing in surrogate models

# This example shows how to define surrogate models with custom model parameters.
# It also shows the validations that are done and how to specify these parameters through
# a configuration.

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.

### Necessary imports

import numpy as np

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
)
from baybe.recommenders import (
    BotorchRecommender,
    FPSRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.surrogates import NGBoostSurrogate
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import add_fake_measurements

### Experiment Setup

parameters = [
    CategoricalParameter(
        name="Granularity",
        values=["coarse", "medium", "fine"],
        encoding="OHE",
    ),
    NumericalDiscreteParameter(
        name="Pressure[bar]",
        values=[1, 5, 10],
        tolerance=0.2,
    ),
    NumericalDiscreteParameter(
        name="Temperature[degree_C]",
        values=np.linspace(100, 200, 10),
    ),
    SubstanceParameter(
        name="Solvent",
        data={
            "Solvent A": "COC",
            "Solvent B": "CCC",
            "Solvent C": "O",
            "Solvent D": "CS(=O)C",
        },
        encoding="MORDRED",
    ),
]

### Create a surrogate model with custom model parameters

# Please note that model_params is an optional argument:
# The defaults will be used if none specified

surrogate_model = NGBoostSurrogate(model_params={"n_estimators": 50, "verbose": True})

### Validation of model parameters

try:
    invalid_surrogate_model = NGBoostSurrogate(model_params={"NOT_A_PARAM": None})
except ValueError as e:
    print("The validator will give an error here:")
    print(e)

### Links for documentation

# [`RandomForestModel`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
# [`NGBoostModel`](https://stanfordmlgroup.github.io/ngboost/1-useage.html)
# [`BayesianLinearModel`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html)

### Creating the campaign

campaign = Campaign(
    searchspace=SearchSpace.from_product(parameters=parameters, constraints=None),
    objective=SingleTargetObjective(target=NumericalTarget(name="Yield", mode="MAX")),
    recommender=TwoPhaseMetaRecommender(
        recommender=BotorchRecommender(surrogate_model=surrogate_model),
        initial_recommender=FPSRecommender(),
    ),
)

### Iterate with recommendations and measurements

# We can print the surrogate model object

print("The model object in json format:")
print(surrogate_model.to_json(), end="\n" * 3)

# Let's do a first round of recommendation
recommendation = campaign.recommend(batch_size=1)

print("Recommendation from campaign:")
print(recommendation)

# Add some fake results
add_fake_measurements(recommendation, campaign.targets)
campaign.add_measurements(recommendation)

### Model Outputs

# Note that this model is only triggered when there is data.

print("Here you will see some model outputs as we set verbose to True")

# Do another round of recommendation
recommendation = campaign.recommend(batch_size=1)


# Print second round of recommendation

print("Recommendation from campaign:")
print(recommendation)

### Using configuration instead

# Note that this can be placed inside an overall campaign config
# Refer to [`create_from_config`](./../Serialization/create_from_config.md) for an example

# Note that the following explicit call `str()` is not strictly necessary.
# It is included since our method of converting this example to a markdown file does not interpret
# this part of the code as `python` code if we do not include this call.

CONFIG = str(
    """
{
    "type": "NGBoostSurrogate",
    "model_params": {
        "n_estimators": 50,
        "verbose": true
    }
}
"""
)

### Model creation from json
recreate_model = NGBoostSurrogate.from_json(CONFIG)

# This configuration creates the same model

assert recreate_model == surrogate_model
