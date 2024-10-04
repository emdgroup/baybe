## Example for Modelling a Slot-Based Mixture

# Explanation

### Imports

from baybe import Campaign
from baybe.constraints import ContinuousLinearConstraint
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace, SubspaceContinuous
from baybe.targets import NumericalTarget

# List of substance labels, divided into subgroups

g1 = ["A", "B"]
g2 = ["mol1", "mol2"]
g3 = ["substance1", "substance2"]

# Make continuous concentration parameters for each group

p_g1_concentrations = [
    NumericalContinuousParameter(name=f"{name}", bounds=(0, 20)) for name in g1
]
p_g2_concentrations = [
    NumericalContinuousParameter(name=f"{name}", bounds=(0, 40)) for name in g2
]
p_g3_concentrations = [
    NumericalContinuousParameter(name=f"{name}", bounds=(0, 60)) for name in g3
]

# Ensure total sum is 100

c_total_sum = ContinuousLinearConstraint(
    parameters=g1 + g2 + g3,
    operator="=",
    coefficients=[1.0] * len(g1 + g2 + g3),
    rhs=100.0,
)

# Ensure sum of group 1 is smaller than 40

c_g1_max = ContinuousLinearConstraint(
    parameters=g1,
    operator="<=",
    coefficients=[1.0] * len(g1),
    rhs=40,
)

# Ensure sum of group 2 is larger than 60

c_g2_min = ContinuousLinearConstraint(
    parameters=g2,
    operator=">=",
    coefficients=[1.0] * len(g2),
    rhs=60,
)

# Create the Campaign
searchspace = SearchSpace(
    continuous=SubspaceContinuous(
        parameters=p_g1_concentrations + p_g2_concentrations + p_g3_concentrations,
        constraints_lin_eq=[c_total_sum],
        constraints_lin_ineq=[c_g1_max, c_g2_min],
    ),
)
campaign = Campaign(
    searchspace=searchspace,
    objective=NumericalTarget(name="MyTarget", mode="MAX").to_objective(),
)

#### Look at some recommendations

# We can quickly verify that the constraints imposed above are respected

rec = campaign.recommend(10)
print(rec)
