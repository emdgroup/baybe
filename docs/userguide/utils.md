# Utilities

BayBE comes with a set of useful functions that can make your life easier in certain
scenarios.

## Search Space Memory Estimation
In search spaces that have discrete parts, the memory needed to store the respective
data can become excessively large as the number of points grows with the amount of
possible combinations arising form all discrete parameter values.

The [`SearchSpace.estimate_product_space_size`](baybe.searchspace.core.SearchSpace.estimate_product_space_size)
and [`SubspaceDiscrete.estimate_product_space_size`](baybe.searchspace.discrete.SubspaceDiscrete.estimate_product_space_size)
utilities allows estimating the memory needed to represent the discrete subspace. 
It will return a [`MemorySize`](baybe.searchspace.discrete.MemorySize) object that
contains some relevant estimates.

```python
import numpy as np

from baybe.parameters import NumericalDiscreteParameter
from baybe.searchspace import SearchSpace

# This will create 10 parameters with 20 values each
# The resulting space would have 20^10 entries, requiring around 745 TB of memory for
# both experimental and computational representation of the search space
parameters = [
    NumericalDiscreteParameter(name=f"p{k+1}", values=np.linspace(0, 100, 20))
    for k in range(10)
]

# Estimate the required memory for such a space in Bytes
mem_estimate = SearchSpace.estimate_product_space_size(parameters)

# Print quantities of interest
print("Experimental Representation")
print(f"Estimated size: {mem_estimate.exp_rep_human_readable}")
print(f"Estimated size in Bytes: {mem_estimate.exp_rep_bytes}")
print(f"Expected data frame shape: {mem_estimate.exp_rep_shape}")

print("Computational Representation")
print(f"Estimated size: {mem_estimate.comp_rep_human_readable}")
print(f"Estimated size in Bytes: {mem_estimate.comp_rep_bytes}")
print(f"Expected data frame shape: {mem_estimate.comp_rep_shape}")
```

```{admonition} Estimate with Constraints
:class: warning
`estimate_product_space_size` currently does not include the influence of potential
constraints in your search space as it is generally very hard to incorporate the effect
of arbitrary constraints without actually buidling the entire space. Hence, you should
always **treat the number you get as upper bound** of required memory. This can still be
useful - for instance if your estimate already is several Exabytes, it is unlikely that
most computers would be able to handle the result even if there are constraints present.
```

```{admonition} Influence of Continuous Parameters
:class: info
Continuous parameters fo not influence the size of the discrete search space part.
Hence, they are ignored by the utility.
```

```{admonition} Memory During Optimization
:class: warning
`estimate_product_space_size` only estimates the memory required to handle the search
space. **It does not estimate the memory required during optimization**, which can be
of a similar magnitude, but generally depends on additional factors.
```

```{admonition} Effective Search Space Creation for Mixtures
:class: tip
If you run into issues creating large search spaces, as for instance for mixtures, you
can try to use the [`SubspaceDiscrete.from_simplex`](baybe.searchspace.discrete.SubspaceDiscrete.from_simplex)
constructor. Instead of creating the search space completely before filtering it down
according to the constraints, this constructor includes the main mixture constraint
already during the Cartesian product, requiring substantially less memory overall. In
addition, BayBE can also be installed with an optional `polars` dependency (`pip install
baybe[polars]`) that will utilize the more efficient machinery form polars for handling
of the search space and its constraints.
```

## Reproducibility
In some scenarios, for instance when testing your code setup, it can be useful to fix
the random seeds for all relevant engines to generate reproducible results. BayBE offers
the [`set_random_seed`](baybe.utils.random.set_random_seed) utility for this purpose:

```python
from baybe.utils.random import set_random_seed

# Set the global random seed for all relevant engines
set_random_seed(1337)

# Assuming we have a prepared campaign
campaign.recommend(5)
```

Setting the global random seed can be undesirable if there are other packages in your
setup. For this, BayBE offers [`temporary_seed`](baybe.utils.random.temporary_seed):

```python
from baybe.utils.random import temporary_seed

# Set the random seed for all relevant engines temporarily within the context
with temporary_seed(1337):
    campaign.recommend(5)
```

## Add Fake Target Measurements and Noise
When creating test scripts, it is often useful to try the recommendation loop for a few
iterations. However, this requires some arbitrary target measurements to be set. Instead
of coming up with a custom logic every time, you can use the
[`add_fake_results`](baybe.utils.dataframe.add_fake_results) utility to add fake target
measurements and the [`add_parameter_noise`](baybe.utils.dataframe.add_parameter_noise)
utility to add artificial parameter noise:

```python
from baybe.utils.dataframe import add_fake_results, add_parameter_noise

# Get recommendations
recommendations = campaign.recommend(5)

# Add fake target measurements and artificial parameter noise to the recommendations
# The utilities will modify the data frames inplace
measurements = recommendations.copy()
add_fake_results(measurements, campaign.targets)
add_parameter_noise(measurements, campaign.parameters)

# Now continue the loop, e.g by adding the measurements...
```
