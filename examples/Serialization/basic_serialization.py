## Example for the serialization of a campaign

# This example shows how to serialize and also de-serialize a campaign.
# It demonstrates and shows that the "original" and "new" objects behave the same.

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.

### Necessary imports

import numpy as np

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
)
from baybe.recommenders import (
    BotorchRecommender,
    FPSRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget

### Experiment setup

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
]

### Creating the campaign

campaign = Campaign(
    searchspace=SearchSpace.from_product(parameters=parameters, constraints=None),
    objective=SingleTargetObjective(target=NumericalTarget(name="Yield", mode="MAX")),
    recommender=TwoPhaseMetaRecommender(
        recommender=BotorchRecommender(),
        initial_recommender=FPSRecommender(),
    ),
)

### Serialization and de-serialization

# We begin by printing the original campaign

print("Original object")
print(campaign, end="\n" * 3)

# We next serialize the campaign to JSON.
# This yields a JSON representation in string format.
# Since it is rather complex, we do not print this string here.
# Note: Dataframes are binary-encoded and are hence not human-readable.

string = campaign.to_json()


# Deserialize the JSON string back to an object.

print("Deserialized object")
campaign_recreate = Campaign.from_json(string)
print(campaign_recreate, end="\n" * 3)

# Verify that both objects are equal.

assert campaign == campaign_recreate
print("Passed basic assertion check!")

### Comparing recommendations in both objects

# To further show how serialization affects working with campaigns, we will now
# create and compare some recommendations in both campaigns.

recommendation_orig = campaign.recommend(batch_size=2)
recommendation_recreate = campaign_recreate.recommend(batch_size=2)

print("Recommendation from original object:")
print(recommendation_orig)

print("Recommendation from recreated object:")
print(recommendation_recreate)
