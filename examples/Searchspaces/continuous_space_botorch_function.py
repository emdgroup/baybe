## Example for using a synthetic BoTorch test function in a continuous searchspace

# Example for using the synthetic test functions in a continuous spaces.
# All test functions that are available in BoTorch are also available here and wrapped
# via the `botorch_function_wrapper`.

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.
# Also, there is a large overlap with other examples with regards to using the test function.
# We thus refer to [`discrete_space`](./discrete_space.md) for details on this aspect.


### Necessary imports for this example

from botorch.test_functions import Rastrigin

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.botorch_wrapper import botorch_function_wrapper

### Defining the test function

# See [`discrete_space`](./../Searchspaces/discrete_space.md) for details.

DIMENSION = 4
TestFunctionClass = Rastrigin

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

BOUNDS = TestFunction.bounds
WRAPPED_FUNCTION = botorch_function_wrapper(test_function=TestFunction)

### Creating the searchspace and the objective

# Since the searchspace is continuous, we use `NumericalContinuousParameter`s.
# We use the data of the test function to deduce bounds and number of parameters.

parameters = [
    NumericalContinuousParameter(
        name=f"x_{k+1}",
        bounds=(BOUNDS[0, k], BOUNDS[1, k]),
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
