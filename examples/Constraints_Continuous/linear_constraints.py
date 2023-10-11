### Example for linear constraints in a continuous searchspace
# pylint: disable=missing-module-docstring

# Example for optimizing a synthetic test functions in a continuous spaces with linear
# constraints.
# All test functions that are available in BoTorch are also available here and wrapped
# via the `botorch_function_wrapper`.

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`baybe_object`](./../Basics/baybe_object.md) for a basic example.
# Also, there is a large overlap with other examples with regards to using the test function.
# We thus refer to [`discrete_space`](./discrete_space.md) for details on this aspect.


#### Necessary imports for this example

from baybe import BayBE
from baybe.constraints import (
    ContinuousEqualityConstraint,
    ContinuousInequalityConstraint,
)
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, Objective
from baybe.utils import botorch_function_wrapper

from botorch.test_functions import Rastrigin

#### Defining the test function

# See [`discrete_space`](./../Searchspaces/discrete_space.md) for details.

DIMENSION = 4
TestFunctionClass = Rastrigin

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

BOUNDS = TestFunction.bounds
WRAPPED_FUNCTION = botorch_function_wrapper(test_function=TestFunction)

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

# We model the following constraints:
# `1.0*x_1 + 1.0*x_2 = 1.0`
# `1.0*x_3 - 1.0*x_4 = 2.0`
# `1.0*x_1 + 1.0*x_3 >= 1.0`
# `2.0*x_2 + 3.0*x_4 <= 1.0` which is equivalent to `-2.0*x_2 - 3.0*x_4 >= -1.0`
constraints = [
    ContinuousEqualityConstraint(
        parameters=["x_1", "x_2"], coefficients=[1.0, 1.0], rhs=1.0
    ),
    ContinuousEqualityConstraint(
        parameters=["x_3", "x_4"], coefficients=[1.0, -1.0], rhs=2.0
    ),
    ContinuousInequalityConstraint(
        parameters=["x_1", "x_3"], coefficients=[1.0, 1.0], rhs=1.0
    ),
    ContinuousInequalityConstraint(
        parameters=["x_2", "x_4"], coefficients=[-2.0, -3.0], rhs=-1.0
    ),
]

searchspace = SearchSpace.from_product(parameters=parameters, constraints=constraints)
objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target", mode="MIN")]
)

#### Constructing the BayBE object and performing a recommendation

baybe_obj = BayBE(
    searchspace=searchspace,
    objective=objective,
)

# Get a recommendation for a fixed batched quantity.

BATCH_QUANTITY = 3
N_ITERATIONS = 3

for k in range(N_ITERATIONS):
    recommendation = baybe_obj.recommend(batch_quantity=BATCH_QUANTITY)

    target_values = []
    for index, row in recommendation.iterrows():
        target_values.append(WRAPPED_FUNCTION(*row.to_list()))

    recommendation["Target"] = target_values

    baybe_obj.add_measurements(recommendation)

### Verify the constraints
measurements = baybe_obj.measurements_exp
TOLERANCE = 0.01

# `1.0*x_1 + 1.0*x_2 = 1.0`
print(
    "1.0*x_1 + 1.0*x_2 = 1.0 satisfied in all recommendations? ",
    (1.0 * measurements["x_1"] + 1.0 * measurements["x_2"])
    .sub(1.0)
    .abs()
    .lt(TOLERANCE)
    .all(),
)

# `1.0*x_3 - 1.0*x_4 = 2.0`
print(
    "1.0*x_3 - 1.0*x_4 = 2.0 satisfied in all recommendations? ",
    (1.0 * measurements["x_3"] - 1.0 * measurements["x_4"])
    .sub(2.0)
    .abs()
    .lt(TOLERANCE)
    .all(),
)

# `1.0*x_1 + 1.0*x_3 >= 1.0`
print(
    "1.0*x_1 + 1.0*x_3 >= 1.0 satisfied in all recommendations? ",
    (1.0 * measurements["x_1"] + 1.0 * measurements["x_3"]).ge(1.0 - TOLERANCE).all(),
)

# `2.0*x_2 + 3.0*x_4 <= 1.0`
print(
    "2.0*x_2 + 3.0*x_4 <= 1.0 satisfied in all recommendations? ",
    (2.0 * measurements["x_2"] + 3.0 * measurements["x_4"]).le(1.0 + TOLERANCE).all(),
)
