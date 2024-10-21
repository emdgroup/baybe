## Example for Modelling a Mixture in Traditional Representation

# When modelling mixtures, one is typically confronted with a large set of substances to
# chose from. In the traditional representation, for each of these substance choices we
# would have one single parameter describing the amount of that substance which should
# go into the mixture. Then, there is one overall constraint to ensure that all substance
# amounts sum to 100%. Additionally, there could be more constraints, for instance if
# there are subgroups of substances that have their own constraints.

# In this example, we will create a simple mixture of up to 6 components. There are
# three subgroups of substances: solvents, bases and phase agents.

### Imports

from baybe import Campaign
from baybe.constraints import ContinuousLinearConstraint
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace, SubspaceContinuous
from baybe.targets import NumericalTarget

### Parameters Setup

# Create lists of substance labels, divided into subgroups.

g1 = ["Solvent1", "Solvent2"]
g2 = ["Base1", "Base2"]
g3 = ["PhaseAgent1", "PhaseAgent2"]

# Make continuous parameters for each subtance amount for each group. Here, the maximum
# amount for each substance depends on the group, i.e. we would allow more addition of each
# solvent compared to bases or phase agents.

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

# Ensure total sum is 100%.

c_total_sum = ContinuousLinearConstraint(
    parameters=g1 + g2 + g3,
    operator="=",
    coefficients=[1.0] * len(g1 + g2 + g3),
    rhs=100.0,
)

# Ensure bases make up at least 10% of the mixture.

c_g2_min = ContinuousLinearConstraint(
    parameters=g2,
    operator=">=",
    coefficients=[1.0] * len(g2),
    rhs=10,
)

# Ensure phase agents make up no more than 5%.

c_g3_max = ContinuousLinearConstraint(
    parameters=g3,
    operator="<=",
    coefficients=[1.0] * len(g3),
    rhs=5,
)

### Campaign Setup

searchspace = SearchSpace(
    continuous=SubspaceContinuous.from_product(
        parameters=p_g1_amounts + p_g2_amounts + p_g3_amounts,
        constraints=[c_total_sum, c_g2_min, c_g3_max],
    ),
)
campaign = Campaign(
    searchspace=searchspace,
    objective=NumericalTarget(name="MyTarget", mode="MAX").to_objective(),
)

### Inspect Some Recommendations

# We can quickly verify that the constraints imposed above are respected.

rec = campaign.recommend(10)
print(rec)
