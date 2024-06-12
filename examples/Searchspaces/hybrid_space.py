## Example for using synthetic test functions in hybrid spaces

# This examples shows how to optimize a custom test function in a hybrid searchspace.
# It focuses on the searchspace-related aspects and not on the custom test function.

# This example assumes some basic familiarity with using BayBE and synthetic test functions.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.
# For details on using synthetic test functions, we refer to other examples in this directory.

### Necessary imports for this example

import numpy as np
from botorch.test_functions import Rastrigin

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter, NumericalDiscreteParameter
from baybe.recommenders import NaiveHybridSpaceRecommender, TwoPhaseMetaRecommender
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.botorch_wrapper import botorch_function_wrapper

### Defining the test function and the hybrid dimensions

# See [`discrete_space`](./discrete_space.md) for details on the test function.

DIMENSION = 6

# This examples uses two lists containing the discrete resp. discrete dimensions.
# Together, these should contain the integers `0,1,...,DIMENSION-1`.
# Although this is checked also in this file, you should configure these indices manually
# manually here and verify that the experiment you set up is configured correctly.
# In particular, if the function that you want to use is only available for a fixed
# dimension, then these will be overwritten by distributing the first half of the
# dimension to `DISC_INDICES` and the remaining ones to `CONT_INDICES`.

DISC_INDICES = [0, 1, 2]
CONT_INDICES = [3, 4, 5]

TestFunctionClass = Rastrigin

# This part checks if the test function already has a fixed dimension.
# In that case, we print a warning and replace DIMENSION.

if not hasattr(TestFunctionClass, "dim"):
    TestFunction = TestFunctionClass(dim=DIMENSION)
elif TestFunctionClass().dim == DIMENSION:
    TestFunction = TestFunctionClass()
else:
    print(
        f"\nYou choose a dimension of {DIMENSION} for the test function"
        f"{TestFunctionClass}. However, this function can only be used in "
        f"{TestFunctionClass().dim} dimension, so the provided dimension is replaced. "
        "Also, DISC_INDICES and CONT_INDICES will be re-written."
    )
    TestFunction = TestFunctionClass()
    DIMENSION = TestFunctionClass().dim
    DISC_INDICES = list(range(0, (DIMENSION + 1) // 2))
    CONT_INDICES = list(range((DIMENSION + 1) // 2, DIMENSION))

# This check verifies if the union of the given index sets yields indices matching `DIMENSION`.
# If this fails, then either the intersection between the index sets is not empty or the test
# function has another dimension.
# Note that this might in particular happen for test functions that ignore the `dim` keyword!

if set(CONT_INDICES + DISC_INDICES) != set(range(DIMENSION)):
    raise ValueError(
        "Either the intersection between CONT_IND and DISC_IND is not empty or your "
        "indices do not match."
    )

BOUNDS = TestFunction.bounds
WRAPPED_FUNCTION = botorch_function_wrapper(test_function=TestFunction)

### Constructing the hybrid searchspace

# The following parameter decides how many points each discrete dimension should have.

POINTS_PER_DIM = 3

# Construct the continuous parameters as `NumericContinuous` parameters.

cont_parameters = [
    NumericalContinuousParameter(
        name=f"x_{k+1}",
        bounds=(BOUNDS[0, k], BOUNDS[1, k]),
    )
    for k in CONT_INDICES
]

# Construct the discrete parameters as `NumericalDiscreteParameters`.

disc_parameters = [
    NumericalDiscreteParameter(
        name=f"x_{k+1}",
        values=list(np.linspace(BOUNDS[0, k], BOUNDS[1, k], POINTS_PER_DIM)),
        tolerance=0.01,
    )
    for k in DISC_INDICES
]

searchspace = SearchSpace.from_product(parameters=disc_parameters + cont_parameters)
objective = SingleTargetObjective(target=NumericalTarget(name="Target", mode="MIN"))

### Constructing hybrid recommenders

# Here, we explicitly create a recommender object to use the `NaiveHybridSpaceRecommender`.
# The keywords `disc_recommender` and `cont_recommender` can be used to select different
# recommenders for the corresponding subspaces.
# We use the default choices, which is the `BotorchRecommender`.

hybrid_recommender = TwoPhaseMetaRecommender(recommender=NaiveHybridSpaceRecommender())

### Constructing the campaign and performing a recommendation

campaign = Campaign(
    searchspace=searchspace,
    objective=objective,
    recommender=hybrid_recommender,
)

# Get a recommendation for a fixed batch size.
BATCH_SIZE = 3
recommendation = campaign.recommend(batch_size=BATCH_SIZE)

# Evaluate the test function.
# Note that we need iterate through the rows of the recommendation.
# Furthermore, we need to interpret the row as a list.

target_values = []
for index, row in recommendation.iterrows():
    target_values.append(WRAPPED_FUNCTION(*row.to_list()))

# We add an additional column with the calculated target values.

recommendation["Target"] = target_values

# Here, we inform the campaign about our measurement.

campaign.add_measurements(recommendation)
print("\n\nRecommended experiments with measured values: ")
print(recommendation)
