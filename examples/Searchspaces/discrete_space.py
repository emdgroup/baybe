"""
Example for using the synthetic test functions in discrete spaces.
All test functions that are available in BoTorch are also available here and wrapped
via the BayBEBotorchFunctionWrapper.

For an example on how to use a custom function instead of a wrapped one, we refer to the
continuous_space example.
"""

import numpy as np

from baybe import BayBE
from baybe.parameters import NumericalDiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, Objective
from baybe.utils import botorch_function_wrapper

# Import the desired test function from botorch here
from botorch.test_functions import Branin

# Here, you can choose the dimension  and the actual the test function.
# All BoTorch test functions can be used.
# Note that some test functions are only defined for specific dimensions.
# If the dimension you provide is not available for the given test function, a warning
# will be printed and the dimension is adjusted.
# For details on constructing the BayBE object, we refer to the basic example file.

# Here, the Dimension and the TestFunctionClass are purposel
DIMENSION = 4
TestFunctionClass = Branin

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
# Parameter for controlling the number of points per dimension.
POINTS_PER_DIM = 4

# Since this is the discrete test, we only construct numerical discrete parameters.
# We use that data of the test function to deduce bounds and number of parameters.
parameters = [
    NumericalDiscreteParameter(
        name=f"x_{k+1}",
        values=list(np.linspace(BOUNDS[0, k], BOUNDS[1, k], POINTS_PER_DIM)),
        tolerance=0.01,
    )
    for k in range(DIMENSION)
]

# Construct searchspace, objective and BayBE object.
searchspace = SearchSpace.from_product(parameters=parameters)

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
