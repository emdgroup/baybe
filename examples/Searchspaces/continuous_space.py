## Creating Continuous Search Spaces

# This example illustrates several ways to create continuous spaces space.

### Imports

import numpy as np

from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace, SubspaceContinuous

### Settings

# We begin by defining the continuous parameters that span our space:

DIMENSION = 4
BOUNDS = (-1, 1)

parameters = [
    NumericalContinuousParameter(name=f"x_{k+1}", bounds=BOUNDS)
    for k in range(DIMENSION)
]

# From these parameter objects, we can now construct a continuous subspace.
# Let us draw some samples from it and verify that they are within the bounds:

subspace = SubspaceContinuous(parameters)
samples = subspace.sample_uniform(10)
print(samples)
assert np.all(samples >= BOUNDS[0]) and np.all(samples <= BOUNDS[1])

# There are several ways we can turn the above objects into a search space.
# This provides a lot of flexibility depending on the context:

# Using conversion:
searchspace1 = SubspaceContinuous(parameters).to_searchspace()

# Explicit attribute assignment via the regular search space constructor:
searchspace2 = SearchSpace(continuous=SubspaceContinuous(parameters))

# Using an alternative search space constructor:
searchspace3 = SearchSpace.from_product(parameters=parameters)


# No matter which version we choose, we can be sure that the resulting search space
# objects are equivalent:

assert searchspace1 == searchspace2 == searchspace3
