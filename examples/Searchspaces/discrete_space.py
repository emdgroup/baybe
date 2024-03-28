## Example for using a synthetic BoTorch test function in a discrete searchspace

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.

### Necessary imports for this example

import numpy as np
from botorch.test_functions import Rastrigin

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.botorch_wrapper import botorch_function_wrapper

### Defining the test function

# BoTorch offers a variety of different test functions, all of which can be used.
# Note that some test functions are only defined for specific dimensions.
# If the dimension you provide is not available for the chose function, a warning will be printed.
# In addition, the dimension is then adjusted automatically.

# Note that choosing a different test function requires to change the `import` statement.
# All test functions that are available in BoTorch are also available here and are later wrapped
# via the `botorch_function_wrapper`.

DIMENSION = 4
TestFunctionClass = Rastrigin

# This code checks if the test function is only available for a specific dimension.
# In that case, we print a warning and replace `DIMENSION`.
# In addition, it constructs the actual `TestFunction` object.

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

# BoTorch provides reasonable bounds for the variables which are used to define the searchspace.

BOUNDS = TestFunction.bounds

# It is necessary to "translate" the BoTorch function such that it can be used by BayBE.
# This is done by using the `botorch_function_wrapper` function.

WRAPPED_FUNCTION = botorch_function_wrapper(test_function=TestFunction)

### Creating the searchspace and the objective

# In this example, we construct a purely discrete space.
# The parameter `POINTS_PER_DIM` controls the number of points per dimension.
# Note that the searchspace will have `POINTS_PER_DIM**DIMENSION` many points.

POINTS_PER_DIM = 4

# Since we have a discrete searchspace, we only construct `NumericalDiscreteParameters`.
# We use the data of the test function to deduce bounds and number of parameters.

parameters = [
    NumericalDiscreteParameter(
        name=f"x_{k+1}",
        values=list(np.linspace(BOUNDS[0, k], BOUNDS[1, k], POINTS_PER_DIM)),
        tolerance=0.01,
    )
    for k in range(DIMENSION)
]

searchspace = SearchSpace.from_product(parameters=parameters)
objective = SingleTargetObjective(target=NumericalTarget(name="Target", mode="MIN"))

### Constructing the campaign and performing a recommendation

campaign = Campaign(
    searchspace=searchspace,
    objective=objective,
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
