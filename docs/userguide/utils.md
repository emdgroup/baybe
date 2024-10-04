# Utilities

BayBE comes with a set of useful functions that can make your life easier in certain
scenarios.

## Search Space Memory Size Estimation
In search spaces that have discrete parts, the memory needed to store the respective
data can become excessively large as the number of points grows with the amount of
possible combinations arising form all discrete parameter values.

The [`estimate_product_space_size`](baybe.searchspace.SearchSpace.estimate_product_space_size)
utility allows estimating the memory needed to represent the discrete subspace. 
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

## Add Fake Target Measurements


