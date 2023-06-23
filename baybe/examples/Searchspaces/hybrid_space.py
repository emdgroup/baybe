"""
Example for using the synthetic test functions in hybrid spaces.
All test functions that are available in BoTorch are also available here and wrapped
via the BayBEBotorchFunctionWrapper.

For an example on how to use a custom function instead of a wrapped one, we refer to the
continuous_space example.
"""

import numpy as np

from baybe.core import BayBE

from baybe.examples.Analytic_Functions.test_functions import BayBEBotorchFunctionWrapper
from baybe.parameters import NumericContinuous, NumericDiscrete
from baybe.searchspace import SearchSpace
from baybe.strategies.bayesian import NaiveHybridRecommender
from baybe.strategies.strategy import Strategy
from baybe.targets import NumericalTarget, Objective

# Import the desired test function from botorch here
from botorch.test_functions import Rastrigin

# Here, you can choose the dimension of the test function and create the actual test
# function. Note that some test functions are only defined for specific dimensions.
# If the dimension you provide is not available for the given test function, a warning
# will be printed and one of the available dimensions is used.
# For details on constructing the baybe object, we refer to the basic example file.
DIMENSION = 6
# DISC_INDICES and CONT_INDICES together should contain the integers 0,1,...,DIMENSION-1
DISC_INDICES = [0, 1, 2]
CONT_INDICES = [3, 4, 5]
TEST_FUNCTION = BayBEBotorchFunctionWrapper(test_function=Rastrigin, dim=DIMENSION)
POINTS_PER_DIM = 3

# This if-statement check whether the union of the given index sets yields indices
# mathcing the dimension of the test function. If this test fails, then either the
# intersection betweem the index sets is not empty or the test function has another
# dimension. Note that this might in particular happen for test functions that ignore
# the dim keyword!
if set(CONT_INDICES + DISC_INDICES) != set(range(TEST_FUNCTION.dim)):
    raise ValueError(
        "Either the tntersection between CONT_IND and DISC_IND is not empty or\
              your indices do not match."
    )

# Construct the continuous parameters as NumericContinuous parameters
cont_parameters = [
    NumericContinuous(
        name=f"x_{k+1}",
        bounds=(TEST_FUNCTION.bounds[0, k], TEST_FUNCTION.bounds[1, k]),
    )
    for k in CONT_INDICES
]

# Construct the discrete parameters as NumericDiscrete parameters
disc_parameters = [
    NumericDiscrete(
        name=f"x_{k+1}",
        values=list(
            np.linspace(
                TEST_FUNCTION.bounds[0, k], TEST_FUNCTION.bounds[1, k], POINTS_PER_DIM
            )
        ),
        tolerance=0.01,
    )
    for k in DISC_INDICES
]

# Construct searchspace, objective and BayBE object.
searchspace = SearchSpace.create(parameters=disc_parameters + cont_parameters)

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
    target_values.append(TEST_FUNCTION(*row.to_list()))
# We add an additional column with the calculated target values...
recommendation["Target"] = target_values
# ... and inform the BayBE object about our measurement.
baybe_obj.add_measurements(recommendation)
print("\n\nRecommended experiments with measured values: ")
print(recommendation)
