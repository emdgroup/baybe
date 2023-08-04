"""
Example for using the synthetic test functions in hybrid spaces.
All test functions that are available in BoTorch are also available here and wrapped
via the BayBEBotorchFunctionWrapper.

For an example on how to use a custom function instead of a wrapped one, we refer to the
continuous_space example.
"""

import numpy as np

from baybe.core import BayBE
from baybe.parameters import NumericalContinuousParameter, NumericalDiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.strategies.bayesian import NaiveHybridRecommender
from baybe.strategies.strategy import Strategy
from baybe.targets import NumericalTarget, Objective

from baybe.utils.botorch_wrapper import botorch_function_wrapper

# Import the desired test function from botorch here
from botorch.test_functions import Rastrigin

# Here, you can choose the dimension  and the actual the test function.
# All BoTorch test functions can be used.
# Note that some test functions are only defined for specific dimensions.
# If the dimension you provide is not available for the given test function, a warning
# will be printed and the dimension is adjusted.
# For details on constructing the BayBE object, we refer to the basic example file.
DIMENSION = 6

# DISC_INDICES and CONT_INDICES together should contain the integers 0,1,...,DIMENSION-1
# NOTE Although this is checked also in this file, you should configure these indices
# manually here and verify that the experiment you set up is configured correctly.
# In particular, if the function that you want to use is only available for a fixed
# dimension, then these will be overwritten by distributing the first half of the
# dimension to DISC_INDICES and the remaining ones to CONT_INDICES
DISC_INDICES = [0, 1, 2]
CONT_INDICES = [3, 4, 5]

TestFunctionClass = Rastrigin

# This part checks if the test function already has a fixed dimension.
# In that case, we print a warning and replace DIMENSION.
if not hasattr(TestFunctionClass, "dim"):
    TestFunction = TestFunctionClass(dim=DIMENSION)  # pylint: disable = E1123
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

# Get the bounds of the variables as they are set by BoTorch
BOUNDS = TestFunction.bounds
# Create the wrapped function itself.
WRAPPED_FUNCTION = botorch_function_wrapper(test_function=TestFunction)

POINTS_PER_DIM = 3

# This if-statement check whether the union of the given index sets yields indices
# matching the dimension of the test function. If this test fails, then either the
# intersection between the index sets is not empty or the test function has another
# dimension. Note that this might in particular happen for test functions that ignore
# the dim keyword!
if set(CONT_INDICES + DISC_INDICES) != set(range(DIMENSION)):
    raise ValueError(
        "Either the intersection between CONT_IND and DISC_IND is not empty or your "
        "indices do not match."
    )

# Construct the continuous parameters
cont_parameters = [
    NumericalContinuousParameter(
        name=f"x_{k+1}",
        bounds=(BOUNDS[0, k], BOUNDS[1, k]),
    )
    for k in CONT_INDICES
]

# Construct the discrete parameters
disc_parameters = [
    NumericalDiscreteParameter(
        name=f"x_{k+1}",
        values=list(np.linspace(BOUNDS[0, k], BOUNDS[1, k], POINTS_PER_DIM)),
        tolerance=0.01,
    )
    for k in DISC_INDICES
]

# Construct searchspace, objective and BayBE object.
searchspace = SearchSpace.from_product(parameters=disc_parameters + cont_parameters)

objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target", mode="MIN")]
)

# Here, we explicitly create a strategy object to use the NaiveHybridRecommender.
# Using the keywords disc_recommender and cont_recommender, it can be chosen which
# individual recommenders should then be used for the corresponding subspaces.
# We use the default choices, which is the SequentialGreedy.

hybrid_recommender = NaiveHybridRecommender()

baybe_obj = BayBE(
    searchspace=searchspace,
    objective=objective,
    strategy=Strategy(recommender=hybrid_recommender),
)


# Get a recommendation for a fixed batched quantity
BATCH_QUANTITY = 3
recommendation = baybe_obj.recommend(batch_quantity=BATCH_QUANTITY)
# Evaluate the test function. Note that we need iterate through the rows of the
# recommendation and that we need to interpret the row as a list.
target_values = []
for index, row in recommendation.iterrows():
    target_values.append(WRAPPED_FUNCTION(*row.to_list()))

# We add an additional column with the calculated target values...
recommendation["Target"] = target_values

# ... and inform the BayBE object about our measurement.
baybe_obj.add_measurements(recommendation)
print("\n\nRecommended experiments with measured values: ")
print(recommendation)
