"""Example for using the synthetic test functions in hybrid spaces."""

import numpy as np

from baybe.core import BayBE
from baybe.parameters import NumericContinuous, NumericDiscrete
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, Objective
from baybe.utils import to_tensor

from test_functions import (
    AckleyTestFunction,
    #    BraninTestFunction,
    #    HartmannTestFunction,
    #    RastriginTestFunction,
    # RosenbrockTestFunction,
    #    ShekelTestFunction,
)

# Here, you can choose the dimension of the test function as well as the actual test
# function. All of the functions that are part of the import statement are available.
# Note that choosing a dimension is only Ackley and Rastrigin, Rosenbrock and will be
# ignored for all other.
# We refer to the test_functions.py file for more information on the functions.
# For details on constructing the baybe object, we refer to the basic example file.
DIMENSION = 6
# DISC_INDICES and CONT_INDICES together should contain the integers 0,1,...,DIMENSION-1
DISC_INDICES = [0, 1, 2]
CONT_INDICES = [3, 4, 5]
TEST_FUNCTION = AckleyTestFunction(dim=DIMENSION)
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

baybe_obj = BayBE(
    searchspace=searchspace,
    objective=objective,
)


# Get a recommendation
recommendation = baybe_obj.recommend(batch_quantity=3)
# Evaluate the test function. Note that we need to transform the recommendation, which
# is a pandas dataframe, to a tensor.
target_value = TEST_FUNCTION(to_tensor(recommendation))
# We add an additional column with the calculated target values...
recommendation["Target"] = target_value
# ... and inform the BayBE object about our measurement.
baybe_obj.add_measurements(recommendation)
print("\n\nRecommended experiments with measured values: ")
print(recommendation)
