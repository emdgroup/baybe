## Example for using a custom BoTorch test function in a continuous searchspace

# This example shows how an arbitrary python function can be used as lookup.

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.

### Necessary imports

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget

### Defining the custom test function

# The function should accept an arbitrary or fixed amount of floats as input.
# It needs to return either a single float or a tuple of floats.
# It is assumed that the analytical test function does only perform a single calculation.
# That is, it is assumed to work in a non-batched-way!

# In this example, we implement a simple sum of squares function with a single output.


def sum_of_squares(*x: float) -> float:
    """Calculate the sum of squares."""
    res = 0
    for y in x:
        res += y**2
    return res


TEST_FUNCTION = sum_of_squares

# For our actual experiment, we need to specify the number of dimension that we want to use.
# This is necessary to know for the creation of the parameters.
# Similarly, it is necessary to state the bounds of the parameters.
# These should be provided as a list of two-dimensional tuples.

DIMENSION = 4
BOUNDS = [(-2, 2), (-2, 2), (-2, 2), (-2, 2)]

### Creating the searchspace and the objective

parameters = [
    NumericalContinuousParameter(
        name=f"x_{k+1}",
        bounds=BOUNDS[k],
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
    target_values.append(TEST_FUNCTION(*row.to_list()))

# We add an additional column with the calculated target values.

recommendation["Target"] = target_values

# Here, we inform the campaign about our measurement.

campaign.add_measurements(recommendation)
print("\n\nRecommended experiments with measured values: ")
print(recommendation)
