## Example for linear constraints in a continuous searchspace

# Example for optimizing a synthetic test functions in a continuous space with linear
# constraints.
# All test functions that are available in BoTorch are also available here.
# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.
# Also, there is a large overlap with other examples with regards to using the test function.
# We thus refer to [`discrete_space`](./../Searchspaces/discrete_space.md) for
# details on this aspect.

### Necessary imports for this example

import os

import numpy as np
import pandas as pd
from botorch.test_functions import Rastrigin

from baybe import Campaign
from baybe.constraints import ContinuousLinearConstraint
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import arrays_to_dataframes

### Defining the test function

# See [`discrete_space`](./../Searchspaces/discrete_space.md) for details.

DIMENSION = 4
TestFunctionClass = Rastrigin

if not hasattr(TestFunctionClass, "dim"):
    TestFunction = TestFunctionClass(dim=DIMENSION)
else:
    TestFunction = TestFunctionClass()
    DIMENSION = TestFunctionClass().dim

BOUNDS = TestFunction.bounds

### Creating the searchspace and the objective

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
# - $1.0*x_1 + 1.0*x_2 = 1.0$
# - $1.0*x_3 - 1.0*x_4 = 2.0$
# - $1.0*x_1 + 1.0*x_3 >= 1.0$
# - $2.0*x_2 + 3.0*x_4 <= 1.0$ which is equivalent to $-2.0*x_2 - 3.0*x_4 >= -1.0$

constraints = [
    ContinuousLinearConstraint(
        parameters=["x_1", "x_2"], operator="=", coefficients=[1.0, 1.0], rhs=1.0
    ),
    ContinuousLinearConstraint(
        parameters=["x_3", "x_4"], operator="=", coefficients=[1.0, -1.0], rhs=2.0
    ),
    ContinuousLinearConstraint(
        parameters=["x_1", "x_3"], operator=">=", coefficients=[1.0, 1.0], rhs=1.0
    ),
    ContinuousLinearConstraint(
        parameters=["x_2", "x_4"], operator="<=", coefficients=[2.0, 3.0], rhs=-1.0
    ),
]

searchspace = SearchSpace.from_product(parameters=parameters, constraints=constraints)
target = NumericalTarget(name="Target", mode="MIN")
objective = target.to_objective()

### Wrap the test function as a dataframe-based lookup callable

lookup = arrays_to_dataframes(
    [p.name for p in parameters], [target.name], use_torch=True
)(TestFunction)

### Construct the campaign and run some iterations

campaign = Campaign(
    searchspace=searchspace,
    objective=objective,
)

# Improve running time for CI via SMOKE_TEST

SMOKE_TEST = "SMOKE_TEST" in os.environ

BATCH_SIZE = 2 if SMOKE_TEST else 3
N_ITERATIONS = 2 if SMOKE_TEST else 3

for k in range(N_ITERATIONS):
    recommendation = campaign.recommend(batch_size=BATCH_SIZE)
    lookup_values = lookup(recommendation)
    measurements = pd.concat([recommendation, lookup_values], axis=1)
    campaign.add_measurements(measurements)

### Verify the constraints

measurements = campaign.measurements
TOLERANCE = 0.01

# $1.0*x_1 + 1.0*x_2 = 1.0$

print(
    "1.0*x_1 + 1.0*x_2 = 1.0 satisfied in all recommendations? ",
    np.allclose(
        1.0 * measurements["x_1"] + 1.0 * measurements["x_2"], 1.0, atol=TOLERANCE
    ),
)

# $1.0*x_3 - 1.0*x_4 = 2.0$

print(
    "1.0*x_3 - 1.0*x_4 = 2.0 satisfied in all recommendations? ",
    np.allclose(
        1.0 * measurements["x_3"] - 1.0 * measurements["x_4"], 2.0, atol=TOLERANCE
    ),
)

# $1.0*x_1 + 1.0*x_3 >= 1.0$

print(
    "1.0*x_1 + 1.0*x_3 >= 1.0 satisfied in all recommendations? ",
    (1.0 * measurements["x_1"] + 1.0 * measurements["x_3"]).ge(1.0 - TOLERANCE).all(),
)

# $2.0*x_2 + 3.0*x_4 <= 1.0$

print(
    "2.0*x_2 + 3.0*x_4 <= 1.0 satisfied in all recommendations? ",
    (2.0 * measurements["x_2"] + 3.0 * measurements["x_4"]).le(1.0 + TOLERANCE).all(),
)


### Using interpoint constraints

# It is also possible to require interpoint constraints which constraint the value of
# a single parameter across a full batch.
# Since these constraints require information about the batch size, they are not used
# during the creation of the search space but handed over to the `recommend` call.
# This example models the following interpoint constraints and combines them also
# with regular constraints.
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

inter_searchspace = SearchSpace.from_product(
    parameters=parameters, constraints=inter_constraints
)

inter_campaign = Campaign(
    searchspace=inter_searchspace,
    objective=objective,
)

for k in range(N_ITERATIONS):
    rec = inter_campaign.recommend(batch_size=BATCH_SIZE)

    # target value are looked up via the botorch wrapper
    target_values = []
    for index, row in rec.iterrows():
        target_values.append(WRAPPED_FUNCTION(*row.to_list()))

    rec["Target"] = target_values
    inter_campaign.add_measurements(rec)
    # Check interpoint constraints
    assert rec["x_1"].sum() >= 2.5 - TOLERANCE
    assert np.isclose(rec["x_2"].sum(), 5)
    assert 2 * rec["x_3"].sum() - rec["x_4"].sum() >= 2.5 - TOLERANCE
