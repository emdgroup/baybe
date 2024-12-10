## Modeling a Mixture in Traditional Representation

# When modeling mixtures, we are often faced with a large set of ingredients to choose
# from. A common way to formalize this type of selection problem is to assign each
# ingredient its own numerical parameter representing the amount of the ingredient in
# the mixture. A sum constraint imposed on all parameters then ensures that the total
# amount of ingredients in the mix is always 100%. In addition, there could be other
# constraints, for instance, to impose further restrictions on individual subgroups of
# ingredients. In BayBE's language, we call this the *traditional mixture
# representation*.

# In this example, we demonstrate how to create a search space in this representation,
# using a simple mixture of up to six components, which are divided into three
# subgroups: solvents, bases and phase agents.

# ```{admonition} Slot-based Representation
# :class: seealso
# For an alternative way to describe mixtures, see our
# [slot-based representation](/examples/Mixtures/slot_based.md).
# ```

### Imports

import numpy as np
import pandas as pd

from baybe.constraints import ContinuousLinearConstraint
from baybe.parameters import NumericalContinuousParameter
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace

### Parameter Setup

# We start by creating lists containing our substance labels according to their
# subgroups:

g1 = ["Solvent1", "Solvent2"]
g2 = ["Base1", "Base2"]
g3 = ["PhaseAgent1", "PhaseAgent2"]

# Next, we create continuous parameters describing the substance amounts for each group.
# Here, the maximum amount for each substance depends on its group, i.e. we allow
# adding more of a solvent compared to a base or a phase agent:

p_g1_amounts = [
    NumericalContinuousParameter(name=f"{name}", bounds=(0, 80)) for name in g1
]
p_g2_amounts = [
    NumericalContinuousParameter(name=f"{name}", bounds=(0, 20)) for name in g2
]
p_g3_amounts = [
    NumericalContinuousParameter(name=f"{name}", bounds=(0, 5)) for name in g3
]

### Constraints Setup

# Now, we set up our constraints. We start with the overall mixture constraint, ensuring
# the total of all ingredients is 100%:

c_total_sum = ContinuousLinearConstraint(
    parameters=g1 + g2 + g3,
    operator="=",
    coefficients=[1] * len(g1 + g2 + g3),
    rhs=100,
)

# Additionally, we require bases make up at least 10% of the mixture:

c_g2_min = ContinuousLinearConstraint(
    parameters=g2,
    operator=">=",
    coefficients=[1] * len(g2),
    rhs=10,
)

# By contrast, phase agents should make up no more than 5%:

c_g3_max = ContinuousLinearConstraint(
    parameters=g3,
    operator="<=",
    coefficients=[1] * len(g3),
    rhs=5,
)

### Search Space Creation

# Having both parameter and constraint definitions at hand, we can create our
# search space:

searchspace = SearchSpace.from_product(
    parameters=[*p_g1_amounts, *p_g2_amounts, *p_g3_amounts],
    constraints=[c_total_sum, c_g2_min, c_g3_max],
)


### Verification of Constraints

# To verify that the constraints imposed above are fulfilled, let us draw some
# random points from the search space:

recommendations = RandomRecommender().recommend(batch_size=10, searchspace=searchspace)
print(recommendations)

# Computing the respective row sums reveals the expected result:

stats = pd.DataFrame(
    {
        "Total": recommendations.sum(axis=1),
        "Total_Bases": recommendations[g2].sum(axis=1),
        "Total_Phase_Agents": recommendations[g3].sum(axis=1),
    }
)
print(stats)

assert np.allclose(stats["Total"], 100)
assert (stats["Total_Bases"] >= 10).all()
assert (stats["Total_Phase_Agents"] <= 5).all()
