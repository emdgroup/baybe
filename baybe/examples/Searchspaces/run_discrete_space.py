"""Example for using the synthetic test functions in discrete spaces."""

import numpy as np

from baybe.core import BayBE

# Note that this import here might be problematic depending on your exact
# setup and that you might need to make some adjustments to make it work!
from baybe.examples.Analytic_Functions.test_functions import (  # pylint: disable=E0401
    #    AckleyTestFunction,
    #    BraninTestFunction,
    #    HartmannTestFunction,
    #    RastriginTestFunction,
    RosenbrockTestFunction,
    #    ShekelTestFunction,
)
from baybe.parameters import NumericDiscrete
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, Objective
from baybe.utils import to_tensor

# Here, you can choose the dimension of the test function as well as the actual test
# function. All of the functions that are part of the import statement are available.
# Note that choosing a dimension is only Ackley and Rastrigin, Rosenbrock and will be
# ignored for all other.
# We refer to the test_functions.py file for more information on the functions.
# For details on constructing the baybe object, we refer to the basic example file.
DIMENSION = 6
TEST_FUNCTION = RosenbrockTestFunction(dim=DIMENSION)
# Parameter for controlling the number of points per dimension.
POINTS_PER_DIM = 4

# Since this is the discrete test, we only construct NumericDiscrete parameters.
# We use that data of the test function to deduce bounds and number of parameters.
parameters = [
    NumericDiscrete(
        name=f"x_{k+1}",
        values=list(
            np.linspace(
                TEST_FUNCTION.bounds[0, k], TEST_FUNCTION.bounds[1, k], POINTS_PER_DIM
            )
        ),
        tolerance=0.01,
    )
    for k in range(TEST_FUNCTION.dim)
]

# Construct searchspace, objective and BayBE object.
searchspace = SearchSpace.create(parameters=parameters)

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
