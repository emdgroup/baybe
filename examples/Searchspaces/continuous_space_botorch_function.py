"""
Example for using synthetic test functions in continuous spaces using a wrapped BoTorch
function as synthetic test function.
"""

from baybe.core import BayBE
from baybe.parameters import NumericContinuous
from baybe.searchspace import SearchSpace
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
DIMENSION = 4
TestFunctionClass = Rastrigin

# This part checks if the test function already has a fixed dimension.
# In that case, we print a warning and replace DIMENSION.
if not hasattr(TestFunctionClass, "dim"):
    TestFunction = TestFunctionClass(dim=DIMENSION)  # pylint: disable = E1123
else:
    print(
        f"\nYou choose a dimension of {DIMENSION} for the test function"
        f"{TestFunctionClass}. However, this function can only be used in "
        f"{TestFunctionClass().dim} dimension, so the provided dimension is replaced."
    )
    TestFunction = TestFunctionClass()
    DIMENSION = TestFunctionClass().dim

# Get the bounds of the variables as they are set by BoTorch
BOUNDS = TestFunction.bounds
# Create the wrapped function itself.
WRAPPED_FUNCTION = botorch_function_wrapper(test_function=TestFunction)

# Since this is the continuous test, we only construct NumericContinuous parameters.
# We use that data of the test function to deduce bounds and number of parameters.
parameters = [
    NumericContinuous(
        name=f"x_{k+1}",
        bounds=(BOUNDS[0, k], BOUNDS[1, k]),
    )
    for k in range(DIMENSION)
]


# For our actual experiment, we need to specify the number of dimension that we want
# to use as this is necessary to know for the creation of the parameters. The same is
# true for the bounds of the parameters which should be provided as a list of
# two-dimensional tuples.

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
