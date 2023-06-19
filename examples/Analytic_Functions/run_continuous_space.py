"""Example for using the synthetic test functions in continuous spaces."""

from baybe.core import BayBE
from baybe.parameters import NumericContinuous
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, Objective
from baybe.utils import to_tensor

from test_functions import (
    #    AckleyTestFunction,
    #    BraninTestFunction,
    #    HartmannTestFunction,
    #    RastriginTestFunction,
    RosenbrockTestFunction,
    #    ShekelTestFunction,
)

# Here, you can choose the dimension of the test function as well as the actual test
# function. All of the functions that are part of the import statement are available.
# Note that choosing a dimension is only Ackley and Rastrigin, Rosenbrock and will be
# ignored for all other.
# We refer to the test_functions.py file for more information on the functions.
# For details on constructing the baybe object, we refer to the basic example file.
DIMENSION = 6
TEST_FUNCTION = RosenbrockTestFunction(dim=DIMENSION)

# Since this is the continuous test, we only construct NumericContinuous parameters.
# We use that data of the test function to deduce bounds and number of parameters.
parameters = [
    NumericContinuous(
        name=f"x_{k+1}",
        bounds=(TEST_FUNCTION.bounds[0, k], TEST_FUNCTION.bounds[1, k]),
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
