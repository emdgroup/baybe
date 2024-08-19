## Example for linear constraints in a continuous searchspace

# Example for optimizing a synthetic test functions in a continuous space with linear
# constraints.
# All test functions that are available in BoTorch are also available here and wrapped
# via the `botorch_function_wrapper`.
# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.
# Also, there is a large overlap with other examples with regards to using the test function.
# We thus refer to [`discrete_space`](./../Searchspaces/discrete_space.md) for
# details on this aspect.

### Necessary imports for this example

import os

import numpy as np
from botorch.test_functions import Rastrigin

from baybe import Campaign
from baybe.constraints import (
    ContinuousInterPointLinearEqualityConstraint,
    ContinuousInterPointLinearInequalityConstraint,
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
)
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
else:
    TestFunction = TestFunctionClass()
    DIMENSION = TestFunctionClass().dim

BOUNDS = TestFunction.bounds
WRAPPED_FUNCTION = botorch_function_wrapper(test_function=TestFunction)

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
    ContinuousLinearEqualityConstraint(
        parameters=["x_1", "x_2"], coefficients=[1.0, 1.0], rhs=1.0
    ),
    ContinuousLinearEqualityConstraint(
        parameters=["x_3", "x_4"], coefficients=[1.0, -1.0], rhs=2.0
    ),
    ContinuousLinearInequalityConstraint(
        parameters=["x_1", "x_3"], coefficients=[1.0, 1.0], rhs=1.0
    ),
    ContinuousLinearInequalityConstraint(
        parameters=["x_2", "x_4"], coefficients=[-2.0, -3.0], rhs=-1.0
    ),
]

searchspace = SearchSpace.from_product(parameters=parameters, constraints=constraints)
objective = SingleTargetObjective(target=NumericalTarget(name="Target", mode="MIN"))

### Construct the campaign and run some iterations

campaign = Campaign(
    searchspace=searchspace,
    objective=objective,
)

# Improve running time for CI via SMOKE_TEST

SMOKE_TEST = "SMOKE_TEST" in os.environ

BATCH_SIZE = 4 if SMOKE_TEST else 5
N_ITERATIONS = 2 if SMOKE_TEST else 3

for k in range(N_ITERATIONS):
    recommendation = campaign.recommend(batch_size=BATCH_SIZE)

    # target value are looked up via the botorch wrapper
    target_values = []
    for index, row in recommendation.iterrows():
        target_values.append(WRAPPED_FUNCTION(*row.to_list()))

    recommendation["Target"] = target_values

    campaign.add_measurements(recommendation)

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


### Using inter-point constraints

# It is also possible to require inter-point constraints which constraint the value of
# a single parameter across a full batch.
# This example models the following inter-point constraints and combines them also
# with regular constraints.
# 1. The sum of `x_1` across the first three batches needs to be at least 2.5.
# 2. The sum of `x_2` across the first four batches needs to be exactly 3.
# 3. The sum of `x_1` and `x_3` across the first three batches needs to be at least 5.

inter_constraints = [
    ContinuousInterPointLinearInequalityConstraint(
        parameters=["x_1"],
        coefficients=[("x_1", 0, 1), ("x_1", 1, 1), ("x_1", 2, 1)],
        rhs=2.5,
    ),
    ContinuousInterPointLinearEqualityConstraint(
        parameters="x_2",  # For single parameters, parameter names are converted.
        coefficients=[("x_2", 0, 1), ("x_2", 1, 1), ("x_2", 2, 1), ("x_2", 3, 1)],
        rhs=3,
    ),
    ContinuousInterPointLinearEqualityConstraint(
        parameters=["x_1", "x_3"],
        coefficients=[
            ("x_1", 0, 1),
            ("x_1", 1, 1),
            ("x_1", 2, 1),
            ("x_3", 0, 1),
            ("x_3", 1, 1),
            ("x_3", 2, 1),
        ],
        rhs=5,
    ),
    ContinuousLinearEqualityConstraint(
        parameters=["x_1", "x_2"], coefficients=[1, 1], rhs=2
    ),
    ContinuousLinearInequalityConstraint(
        parameters=["x_3", "x_4"], coefficients=[-1, 1], rhs=-1
    ),
]

### Re-construct search space and campaign and run some iterations

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
    # Check inter-point constraints
    assert rec.at[0, "x_1"] + rec.at[1, "x_1"] + rec.at[2, "x_1"] >= 2.5 - TOLERANCE
    assert np.isclose(
        rec.at[0, "x_2"] + rec.at[1, "x_2"] + rec.at[2, "x_2"] + rec.at[3, "x_2"],
        3,
    )
    assert (
        rec.at[0, "x_1"]
        + rec.at[1, "x_1"]
        + rec.at[2, "x_1"]
        + rec.at[0, "x_3"]
        + rec.at[1, "x_3"]
        + rec.at[2, "x_3"]
        >= 4 - TOLERANCE
    )
    # Check normal constraints for each batch individually
    for b in range(BATCH_SIZE):
        assert np.isclose(rec.at[b, "x_1"] + rec.at[b, "x_2"], 2)
        assert -rec.at[b, "x_3"] + rec.at[b, "x_4"] >= -1 - TOLERANCE
