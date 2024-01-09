### Example for using a custom BoTorch test function in a continuous searchspace

# This example shows how an arbitrary python function can be used as lookup.

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.

#### Necessary imports

import numpy as np
import pandas as pd

from baybe import Campaign
from baybe.objective import Objective
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget

#### Defining the custom test function

# The function should accept an arbitrary or fixed amount of floats as input.
# It needs to return either a single float or a tuple of floats.
# It is assumed that the analytical test function does only perform a single calculation.
# That is, it is assumed to work in a non-batched-way!

# In this example, we implement a simple sum of squares function with a single output.


def sum_of_squares(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the sum of squares of all parameter values."""
    return np.square(df).sum(axis=1).rename("Target").to_frame()


TEST_FUNCTION = sum_of_squares

# For our actual experiment, we need to specify the number of dimension that we want to use.
# This is necessary to know for the creation of the parameters.
# Similarly, it is necessary to state the bounds of the parameters.
# These should be provided as a list of two-dimensional tuples.

DIMENSION = 4
BOUNDS = [(-2, 2), (-2, 2), (-2, 2), (-2, 2)]

#### Creating the searchspace and the objective

parameters = [
    NumericalContinuousParameter(
        name=f"x_{k+1}",
        bounds=BOUNDS[k],
    )
    for k in range(DIMENSION)
]

searchspace = SearchSpace.from_product(parameters=parameters)

objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target", mode="MIN")]
)

#### Constructing the campaign and performing a recommendation

campaign = Campaign(
    searchspace=searchspace,
    objective=objective,
)

# Get a recommendation for a fixed batched quantity.
BATCH_QUANTITY = 3
recommendation = campaign.recommend(batch_quantity=BATCH_QUANTITY)

# We now evaluate the test function. The target values are then appended to the
# recommendations dataframe.
measurements = pd.concat([recommendation, TEST_FUNCTION(recommendation)], axis=1)

# Lastly, we inform the campaign about our measurement.
campaign.add_measurements(measurements)
print("\n\nRecommended experiments with measured values:")
print(measurements)
