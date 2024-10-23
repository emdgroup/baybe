## Basic example for using BayBE

# This example shows how to create a campaign and how to use it.
# It is intended to be used as a first point of interaction with campaign after having
# read the corresponding [user guide](./../../userguide/campaigns).

### Necessary imports for this example

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, SubstanceParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import add_fake_measurements

### Setup

# This example presents the optimization of a direct Arylation reaction in a discrete
# space. For this, we require data for solvents, ligands and bases.

dict_solvent = {
    "DMAc": r"CC(N(C)C)=O",
    "Butyornitrile": r"CCCC#N",
    "Butyl Ester": r"CCCCOC(C)=O",
    "p-Xylene": r"CC1=CC=C(C)C=C1",
}

dict_base = {
    "Potassium acetate": r"O=C([O-])C.[K+]",
    "Potassium pivalate": r"O=C([O-])C(C)(C)C.[K+]",
    "Cesium acetate": r"O=C([O-])C.[Cs+]",
    "Cesium pivalate": r"O=C([O-])C(C)(C)C.[Cs+]",
}

dict_ligand = {
    "BrettPhos": r"CC(C)C1=CC(C(C)C)=C(C(C(C)C)=C1)C2=C(P(C3CCCCC3)C4CCCCC4)C(OC)="
    "CC=C2OC",
    "Di-tert-butylphenylphosphine": r"CC(C)(C)P(C1=CC=CC=C1)C(C)(C)C",
    "(t-Bu)PhCPhos": r"CN(C)C1=CC=CC(N(C)C)=C1C2=CC=CC=C2P(C(C)(C)C)C3=CC=CC=C3",
}

# We define the chemical substances parameters using the dictionaries defined previously.
# Here, we use `"MORDRED"` encoding, but others are available.
# We proceed to define numerical discrete parameters `temperature` and `concentration`
# and create the search space.

solvent = SubstanceParameter("Solvent", data=dict_solvent, encoding="MORDRED")
base = SubstanceParameter("Base", data=dict_base, encoding="MORDRED")
ligand = SubstanceParameter("Ligand", data=dict_ligand, encoding="MORDRED")

temperature = NumericalDiscreteParameter(
    "Temperature", values=[90, 105, 120], tolerance=2
)
concentration = NumericalDiscreteParameter(
    "Concentration", values=[0.057, 0.1, 0.153], tolerance=0.005
)

parameters = [solvent, base, ligand, temperature, concentration]

searchspace = SearchSpace.from_product(parameters=parameters)

# In this example, we maximize the yield of a reaction and define a corresponding
# objective.

objective = SingleTargetObjective(target=NumericalTarget(name="yield", mode="MAX"))

# We now finally create the campaign using the objects configure previously.

campaign = Campaign(
    searchspace=searchspace,
    objective=objective,
)

### Getting a recommendation and adding measurements

# We use the `recommend()` function of the campaign for getting measurements.

recommendation = campaign.recommend(batch_size=2)

print("\n\nRecommended measurements with batch_size = 2: ")
print(recommendation)

# Adding target values is done by creating a new column in the `recommendation`
# dataframe named after the target.
# In this example, we use the `add_fake_measurements()` utility to create fake results.
# We then update the campaign by adding the measurements.

add_fake_measurements(recommendation, campaign.targets)
print("\n\nRecommended experiments with fake measured values: ")
print(recommendation)

campaign.add_measurements(recommendation)
