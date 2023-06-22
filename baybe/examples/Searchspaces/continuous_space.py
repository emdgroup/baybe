"""
Example for using the synthetic test functions in continuous spaces.
All test functions that are available in BoTorch are also available here and wrapped
via the BayBEBotorchFunctionWrapper.
"""

from baybe.core import BayBE

from baybe.examples.Analytic_Functions.test_functions import BayBEBotorchFunctionWrapper
from baybe.parameters import NumericContinuous
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, Objective
from baybe.utils import to_tensor

# Import the desired test function from botorch here
from botorch.test_functions import Rastrigin

# Here, you can choose the dimension of the test function and create the actual test
# function. Note that some test functions are only defined for specific dimensions.
# If the dimension you provide is not available for the given test function, a warning
# will be printed and one of the available dimensions is used.
# For details on constructing the baybe object, we refer to the basic example file.
DIMENSION = 2
TEST_FUNCTION = BayBEBotorchFunctionWrapper(test_function=Rastrigin, dim=DIMENSION)

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

# Get a recommendation for a fixed batched quantity
BATCH_QUANTITY = 3
recommendation = baybe_obj.recommend(batch_quantity=BATCH_QUANTITY)
# Evaluate the test function. Note that we need to transform the recommendation, which
# is a pandas dataframe, to a tensor.
target_value = TEST_FUNCTION(to_tensor(recommendation))
# We add an additional column with the calculated target values...
recommendation["Target"] = target_value
# ... and inform the BayBE object about our measurement.
baybe_obj.add_measurements(recommendation)
print("\n\nRecommended experiments with measured values: ")
print(recommendation)
