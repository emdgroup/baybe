## Example for linear interpoint constraints in a continuous searchspace

# Example for optimizing a synthetic test functions in a continuous space with linear
# interpoint constraints.
# While intrapoint constraints impose conditions on each individual point of a batch,
# interpoint constraints do so **across** the points of the batch. That is, an
# interpoint constraint of the form ``x_1 + x_2 <= 1`` enforces that the sum of all
# ``x_1`` values plus the sum of all ``x_2`` values in the batch must not exceed 1.

# This example is a variant of the example for linear constraints, and we thus refer
# to [`linear_constraints`](./linear_constraints.md) for more details and explanations.

### Necessary imports for this example

import os

import numpy as np
from botorch.test_functions import Rastrigin

from baybe import Campaign
from baybe.constraints import ContinuousLinearConstraint
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.botorch_wrapper import botorch_function_wrapper

### Defining the test function

DIMENSION = 4
TestFunctionClass = Rastrigin

if not hasattr(TestFunctionClass, "dim"):
    TestFunction = TestFunctionClass(dim=DIMENSION)
else:
    TestFunction = TestFunctionClass()
    DIMENSION = TestFunctionClass().dim

BOUNDS = TestFunction.bounds
WRAPPED_FUNCTION = botorch_function_wrapper(test_function=TestFunction)

### Creating the searchspace and the objective

parameters = [
    NumericalContinuousParameter(
        name=f"x_{k+1}",
        bounds=(BOUNDS[0, k], BOUNDS[1, k]),
    )
    for k in range(DIMENSION)
]

### Defining interpoint constraints

# This example models the following interpoint constraints:
# 1. The sum of `x_1` across all batches needs to be >= 2.5.
# 2. The sum of `x_2` across all batches needs to be exactly 5.
# 3. The sum of `2*x_3` minus the sum of `x_4` across all batches needs to be >= 5.


inter_constraints = [
    ContinuousLinearConstraint(
        parameters=["x_1"], operator=">=", coefficients=[1], rhs=2.5, interpoint=True
    ),
    ContinuousLinearConstraint(
        parameters=["x_2"], operator="=", coefficients=[1], rhs=5, interpoint=True
    ),
    ContinuousLinearConstraint(
        parameters=["x_3", "x_4"],
        operator=">=",
        coefficients=[2, -1],
        rhs=5,
        interpoint=True,
    ),
]

### Construct search space without the previous constraints

searchspace = SearchSpace.from_product(
    parameters=parameters, constraints=inter_constraints
)
objective = SingleTargetObjective(target=NumericalTarget(name="Target", mode="MIN"))

campaign = Campaign(
    searchspace=searchspace,
    objective=objective,
)

# Improve running time for CI via SMOKE_TEST

SMOKE_TEST = "SMOKE_TEST" in os.environ

BATCH_SIZE = 4 if SMOKE_TEST else 5
N_ITERATIONS = 2 if SMOKE_TEST else 3
TOLERANCE = 0.01

for k in range(N_ITERATIONS):
    rec = campaign.recommend(batch_size=BATCH_SIZE)

    target_values = []
    for index, row in rec.iterrows():
        target_values.append(WRAPPED_FUNCTION(*row.to_list()))

    rec["Target"] = target_values
    campaign.add_measurements(rec)

    # Check interpoint constraints

    print(
        "The sum of `x_1` across all batches is at least >= 2.5",
        rec["x_1"].sum() >= 2.5 - TOLERANCE,
    )
    print(
        "The sum of `x_2` across all batches is exactly 5",
        np.isclose(rec["x_2"].sum(), 5),
    )
    print(
        "The sum of `2*x_3` minus the sum of `x_4` across all batches is at least >= 5",
        2 * rec["x_3"].sum() - rec["x_4"].sum() >= 2.5 - TOLERANCE,
    )