### Example for using a synthetic BoTorch test function in a continuous searchspace
import pandas as pd

# Example for using the synthetic test functions in a continuous spaces.
# All test functions that are available in BoTorch can be used here.
# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.
# Also, there is a large overlap with other examples with regards to using the test function.
# We thus refer to [`discrete_space`](./discrete_space.md) for details on this aspect.
#### Necessary imports for this example
from botorch.test_functions import Rastrigin

from baybe import Campaign
from baybe.objective import Objective
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils import add_dataframe_layer

#### Defining the test function

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

#### Creating the searchspace and the objective

# Since the searchspace is continuous test, we construct `NumericalContinuousParameter`s
# We use that data of the test function to deduce bounds and number of parameters.
parameters = [
    NumericalContinuousParameter(
        name=f"x_{k+1}",
        bounds=(BOUNDS[0, k], BOUNDS[1, k]),
    )
    for k in range(DIMENSION)
]

searchspace = SearchSpace.from_product(parameters=parameters)
targets = [NumericalTarget(name="Target", mode="MIN")]
objective = Objective(mode="SINGLE", targets=targets)

#### Constructing the campaign and performing a recommendation

campaign = Campaign(
    searchspace=searchspace,
    objective=objective,
)

# Get a recommendation for a fixed batched quantity.
BATCH_QUANTITY = 3
recommendation = campaign.recommend(batch_quantity=BATCH_QUANTITY)

# We now evaluate the test function. For this purpose, we wrap it such that it
# supports dataframe inputs and outputs. The target values are then appended to the
# recommendations dataframe.
WRAPPED_FUNCTION = add_dataframe_layer(TestFunction, [t.name for t in targets])
measurements = pd.concat([recommendation, WRAPPED_FUNCTION(recommendation)], axis=1)

# Lastly, we inform the campaign about our measurement.
campaign.add_measurements(measurements)
print("\n\nRecommended experiments with measured values:")
print(measurements)
