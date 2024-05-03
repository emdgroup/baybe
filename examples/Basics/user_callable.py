## Example of a user-defined callable

# This example is an extension of the recommender example.
# It implements an end-to-end use case demonstrating how to utilize the user-defined callable in pure recommenders.

# This examples assumes some basic familiarity with using BayBE.
# We refer to [`campaign`](./campaign.md) for a more general and basic example.

### Necessary imports for this example

from typing import Optional

import numpy as np
from pandas import DataFrame

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import CategoricalParameter, NumericalDiscreteParameter
from baybe.recommenders import (
    RandomRecommender,
    SequentialGreedyRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import add_fake_results

### Callable function

# This function prints all given parameters and pickles the recommender object to test its serializability.


def callable_test(
    recommender: Optional[SequentialGreedyRecommender] = None,
    searchspace: Optional[SearchSpace] = None,
    batch_size: Optional[int] = None,
    train_x: Optional[DataFrame] = None,
    train_y: Optional[DataFrame] = None,
):
    """Test the functionality of the implemented callable attribute.

    Args:
        recommender: Recommender to be used.
        searchspace: The search space from which to recommend the points.
        batch_size: The number of points to be recommended.
        train_x: Optional training inputs for training a model.
        train_y: Optional training labels for training a model.
    """
    print("Start callable_test")
    print(f"""\ntrain x:\n{train_x}\ntrain y:\n{train_y}""")
    print("End callable_test")


### Creating the recommender object

recommender = TwoPhaseMetaRecommender(
    initial_recommender=RandomRecommender(),
    recommender=SequentialGreedyRecommender(
        user_callable=callable_test,
    ),
)

### Setup Searchspace parameters

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

# We create the searchspace and the objective.

searchspace = SearchSpace.from_product(parameters=parameters)

objective = SingleTargetObjective(target=NumericalTarget(name="yield", mode="MAX"))

### Creating the campaign

# The recommender object can now be used together with the searchspace and the objective as follows.

campaign = Campaign(
    searchspace=searchspace,
    recommender=recommender,
    objective=objective,
)

# We now run some loops of the recommendation process to demonstrate the use of the callable

for i in range(2):
    recommendation = campaign.recommend(batch_size=3)
    print("\n\nRecommended experiments: ")
    print(recommendation)

    add_fake_results(recommendation, campaign)
    print("\n\nRecommended experiments with fake measured values: ")
    print(recommendation)

    campaign.add_measurements(recommendation)
