## Example for linear interpoint constraints in a continuous searchspace

# Example for optimizing a synthetic test functions in a continuous space with linear
# interpoint constraints.
# While intrapoint constraints impose conditions on each individual point of a batch,
# interpoint constraints do so **across** the points of the batch. That is, an
# interpoint constraint of the form ``x_1 + x_2 <= 1`` enforces that the sum of all
# ``x_1`` values plus the sum of all ``x_2`` values in the batch must not exceed 1.
# A possible relevant constraint might be that only 100ml of a given solvent are available for
# a full batch, but there is no limit for the amount of solvent to use for a single experiment
# within that batch.

# This example is a variant of the example for linear constraints, and we thus refer
# to [`linear_constraints`](./linear_constraints.md) for more details and explanations.

### Necessary imports for this example


import pandas as pd
from botorch.test_functions import Rastrigin

from baybe import Campaign
from baybe.constraints import ContinuousLinearConstraint
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import arrays_to_dataframes

### Defining the test function

DIMENSION = 4
test_function = Rastrigin(dim=DIMENSION)
BOUNDS = test_function.bounds

### Creating the searchspace and the objective

parameters = [
    NumericalContinuousParameter(
        name=f"x_{k + 1}",
        bounds=(BOUNDS[0, k], BOUNDS[1, k]),
    )
    for k in range(DIMENSION)
]

### Defining interpoint constraints

# This example models the following interpoint constraints:
# 1. The sum of `x_1` across all batches needs to be >= 2.5.
# 2. The sum of `x_2` across all batches needs to be exactly 5.
# 3. The sum of `2*x_3` minus the sum of `x_4` across all batches needs to be >= 2.5.


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
        rhs=2.5,
        interpoint=True,
    ),
]

### Construct search space without the previous constraints

searchspace = SearchSpace.from_product(
    parameters=parameters, constraints=inter_constraints
)
target = NumericalTarget(name="Target", mode="MIN")
objective = target.to_objective()

### Wrap the test function as a dataframe-based lookup callable

lookup = arrays_to_dataframes(
    [p.name for p in parameters], [target.name], use_torch=True
)(test_function)

campaign = Campaign(
    searchspace=searchspace,
    objective=objective,
)

BATCH_SIZE = 5
N_ITERATIONS = 3
TOLERANCE = 0.01

for k in range(N_ITERATIONS):
    rec = campaign.recommend(batch_size=BATCH_SIZE)
    lookup_values = lookup(rec)
    measurements = pd.concat([rec, lookup_values], axis=1)
    campaign.add_measurements(measurements)

    # Check interpoint constraints

    print(
        "The sum of `x_1` across all batches is at least >= 2.5",
        rec["x_1"].sum() >= 2.5 - TOLERANCE,
    )
    print(
        "The sum of `x_2` across all batches is exactly 5",
        abs(rec["x_2"].sum() - 5) < TOLERANCE,
    )
    print(
        "The sum of `2*x_3` minus the sum of `x_4` across all batches is at least >= 2.5",
        2 * rec["x_3"].sum() - rec["x_4"].sum() >= 2.5 - TOLERANCE,
    )
