"""
Example for using synthetic test functions in continuous spaces.
A synthetic test function can either be a wrapped BoTorch test function or a custom
test function.
"""

from baybe.core import BayBE

from baybe.examples.Analytic_Functions.test_functions import BayBEBotorchFunctionWrapper
from baybe.parameters import NumericContinuous
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, Objective

# Import the desired test function from botorch here
from botorch.test_functions import Rastrigin

# CASE 1: USING A BOTORCH TEST FUNCTION

# Here, you can choose the dimension of the test function and create the actual test
# function. Note that some test functions are only defined for specific dimensions.
# If the dimension you provide is not available for the given test function, a warning
# will be printed and one of the available dimensions is used.
# For details on constructing the baybe object, we refer to the basic example file.
DIMENSION = 5
TEST_FUNCTION_BOTORCH = BayBEBotorchFunctionWrapper(
    test_function=Rastrigin, dim=DIMENSION
)

# Since this is the continuous test, we only construct NumericContinuous parameters.
# We use that data of the test function to deduce bounds and number of parameters.
parameters_botorch = [
    NumericContinuous(
        name=f"x_{k+1}",
        bounds=(TEST_FUNCTION_BOTORCH.bounds[0, k], TEST_FUNCTION_BOTORCH.bounds[1, k]),
    )
    for k in range(TEST_FUNCTION_BOTORCH.dim)
]

# CASE 2: USING A CUSTOM TEST FUNCTION


# Here, we now need to define our custom function first. The function should accept an
# arbitrary or fixed amount of floats as input and return either a single float or
# a tuple of floats.
# NOTE It is assumed that the analytical test function does only perform a single
# calculation, i.e., it is assumed to work in a non-batched-way!
# NOTE The case of using an analytical test function for multi-target optimization has
# not yet been fully tested.
# We implements a simple sum of squares function with a single output.
def sum_of_squares(*x: float) -> float:
    """
    Calculates the sum of squares.
    """
    res = 0
    for y in x:
        res += y**2
    return res


# For our actual experiment, we need to specify the number of dimension that we want
# to use as this is necessary to know for the creation of the parameters. The same is
# true for the bounds of the parameters which should be provided as a list of
# two-dimensional tuples.

TEST_FUNCTION_CUSTOM = sum_of_squares

DIMENSION = 4
BOUNDS = [(-2, 2), (-2, 2), (-2, 2), (-2, 2)]

parameters_custom = [
    NumericContinuous(
        name=f"x_{k+1}",
        bounds=BOUNDS[k],
    )
    for k in range(DIMENSION)
]


# Construct searchspace, objective and BayBE object.
# Here you need to choose the correct parameters, depending on what you want to test!
searchspace = SearchSpace.create(parameters=parameters_custom)
# searchspace = SearchSpace.create(parameters=parameters_botorch)

objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target", mode="MIN")]
)

baybe_obj = BayBE(
    searchspace=searchspace,
    objective=objective,
)

# Here, we now choose the actual test function.
# Make sure that this matches your previous choice!
# TEST_FUNCTION = TEST_FUNCTION_CUSTOM
TEST_FUNCTION = TEST_FUNCTION_BOTORCH


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
