## Example for a user defined callable

# This example is an extenssion of the recommenders example.
# It implements an End-to-End example, that showes how to use the user defined callable in pure recommenders.

# This examples assumes some basic familiarity with using BayBE.
# We refer to [`campaign`](./campaign.md) for a more general and basic example.

### Necessary imports for this example

import pickle
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
from baybe.surrogates import GaussianProcessSurrogate
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import add_fake_results

# Per default the initial recommender chosen is a random recommender.
INITIAL_RECOMMENDER = RandomRecommender()

### Surrogate models

# This model uses available data to model the objective function as well as the uncertainty.
# The surrogate model is then used by the acquisition function to make recommendations.

# Per default a Gaussian Process is used
SURROGATE_MODEL = GaussianProcessSurrogate()

### Acquisition function

# This function looks for points where measurements of the target value could improve the model.

# Note that the qvailability of the acquisition functions might depend on the `batch_size`:
#   - If `batch_size` is set to 1, all available acquisition functions can be chosen
#   - If a larger value is chosen, only those that allow batching.
#       That is, 'q'-variants of the acquisition functions must be chosen.

# The default acquisition function is q-Expected Improvement.

ACQ_FUNCTION = "qEI"


### Callable function
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

    This function prints all giving parameters and pickles the recommender object
    to test the serializability.
    """
    print("Start callable_test")
    print(
        f"""{recommender}\n{searchspace}\nBatch Size:{batch_size}
    \ntrain x:\n{train_x}\ntrain y:\n{train_y}"""
    )
    with open("callableTest.pkl", "wb") as file:
        pickle.dump(recommender, file)
    print("End callable_test")


### Other parameters

# Two other boolean hyperparameters can be specified when creating a recommender object.
# The first one allows the recommendation of points that were already recommended previously.
# The second one allows the recommendation of points that have already been measured.
# Per default, they are set to `True`.

ALLOW_REPEATED_RECOMMENDATIONS = True
ALLOW_RECOMMENDING_ALREADY_MEASURED = True

### Creating the recommender object

# To create the recommender object, each parameter described above can be specified as follows.
# Note that they all have default values.
# Therefore one does not need to specify all of them to create a recommender object.

recommender = TwoPhaseMetaRecommender(
    initial_recommender=INITIAL_RECOMMENDER,
    recommender=SequentialGreedyRecommender(
        surrogate_model=SURROGATE_MODEL,
        acquisition_function=ACQ_FUNCTION,
        allow_repeated_recommendations=ALLOW_REPEATED_RECOMMENDATIONS,
        allow_recommending_already_measured=ALLOW_RECOMMENDING_ALREADY_MEASURED,
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

# This campaign can then be used to get recommendations and add measurements:
# We use a loop to make sure that the recommend call of the pure recommender is made.
# The initial recommend call is made in the campaign, and from that point on, the recommend call of the pure recommender will be used to train the recommender based on the previous recommendations.
for i in range(2):
    recommendation = campaign.recommend(batch_size=3)
    print("\n\nRecommended experiments: ")
    print(recommendation)

    add_fake_results(recommendation, campaign)
    print("\n\nRecommended experiments with fake measured values: ")
    print(recommendation)

    campaign.add_measurements(recommendation)
