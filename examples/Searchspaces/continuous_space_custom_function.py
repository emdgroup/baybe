"""
Example for using custom synthetic test functions in continuous spaces.
"""

from baybe.core import BayBE
from baybe.parameters import NumericContinuous
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, Objective


# Here, we now need to define our custom function first. The function should accept an
# arbitrary or fixed amount of floats as input and return either a single float or
# a tuple of floats.
# NOTE It is assumed that the analytical test function does only perform a single
# calculation, i.e., it is assumed to work in a non-batched-way!
# We implement a simple sum of squares function with a single output.
def sum_of_squares(*x: float) -> float:
    """
    Calculates the sum of squares.
    """
    res = 0
    for y in x:
        res += y**2
    return res


# For our actual experiment, we need to specify the number of dimension that we want
# to use as this is necessary to know for the creation of the parameters. The same is
# true for the bounds of the parameters which should be provided as a list of
# two-dimensional tuples.

TEST_FUNCTION = sum_of_squares

DIMENSION = 4
BOUNDS = [(-2, 2), (-2, 2), (-2, 2), (-2, 2)]

parameters = [
    NumericContinuous(
        name=f"x_{k+1}",
        bounds=BOUNDS[k],
    )
    for k in range(DIMENSION)
]


# Construct searchspace, objective and BayBE object.
searchspace = SearchSpace.create(parameters=parameters)

objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target", mode="MIN")]
)

baybe_obj = BayBE(
    searchspace=searchspace,
    objective=objective,
)

# Get a recommendation for a fixed batched quantity
BATCH_QUANTITY = 3
recommendation = baybe_obj.recommend(batch_quantity=BATCH_QUANTITY)
# Evaluate the test function. Note that we need iterate through the rows of the
# recommendation and that we need to interpret the row as a list.
target_values = []
for index, row in recommendation.iterrows():
    target_values.append(TEST_FUNCTION(*row.to_list()))

# We add an additional column with the calculated target values...
recommendation["Target"] = target_values

# ... and inform the BayBE object about our measurement.
baybe_obj.add_measurements(recommendation)
print("\n\nRecommended experiments with measured values: ")
print(recommendation)
