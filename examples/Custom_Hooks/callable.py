## E2E Test for using callables

# This example shows how to use the register_hook function to request internal data from different objects such as campaign, recommenders and objective.

# To request internal data from the recommendation process this function can be used by giving it two arguments:
# - The class method the hook should be attached to
# - The callable function that will process the data
# The function wraps the class method and calls the callable function

# This examples assumes some basic familiarity with using BayBE.
# We refer to [`campaign`](./campaign.md) for a more general and basic example.

### Necessary imports for this example


import pandas as pd

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.objectives.base import Objective
from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.basic import register_hook
from baybe.utils.dataframe import add_fake_results

### Setup

# Define three test functions to test the functionality of register_hook().
# Note that the callable function needs to have the same signature as the function it is attached to.


def recommend_hook(
    self,
    batch_size: int,
    searchspace: SearchSpace,
    objective: Objective | None = None,
    measurements: pd.DataFrame | None = None,
):
    """Print the compatibility of the recommender, the searchspace and the batch size from the recommend call."""
    print("start recommend_hook")
    print(self.compatibility)
    print(searchspace, batch_size)
    print("End recommend_hook")


### Example

# We create a two phase meta recommender with default values.

recommender = RandomRecommender()

# We overwrite the original recommend method of the sequential greedy recommender class.

RandomRecommender.recommend = register_hook(RandomRecommender.recommend, recommend_hook)

# We define all needen parameters for this example and collect them in a list.

temperature = NumericalDiscreteParameter(
    "Temperature", values=[90, 105, 120], tolerance=2
)
concentration = NumericalDiscreteParameter(
    "Concentration", values=[0.057, 0.1, 0.153], tolerance=0.005
)
parameters = [temperature, concentration]

# We create the searchspace and the objective.

searchspace = SearchSpace.from_product(parameters=parameters)
objective = SingleTargetObjective(target=NumericalTarget(name="yield", mode="MAX"))

### Creating the campaign

campaign = Campaign(
    searchspace=searchspace,
    recommender=recommender,
    objective=objective,
)

# This campaign can now be used to get recommendations, add measurements and process the recommend_hook:
# Note that a for loop is used to train the data of the second recommendation on the data from the first one.

for _ in range(2):
    recommendation = campaign.recommend(batch_size=3)
    print("\n\nRecommended experiments: ")
    print(recommendation)

    add_fake_results(recommendation, campaign)
    print("\n\nRecommended experiments with fake measured values: ")
    print(recommendation)

    campaign.add_measurements(recommendation)
