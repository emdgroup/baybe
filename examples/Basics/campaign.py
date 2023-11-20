### Basic example for using BayBE

# This example shows how to create a campaign and how to use it.
# It details how a user can first define parameters of the searchspace and the objective.
# These can then be used to create a proper campaign that can be used to get recommendations.

#### Necessary imports for this example

from baybe import Campaign
from baybe.objective import Objective
from baybe.parameters import NumericalDiscreteParameter, SubstanceParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils import add_fake_results

#### Creation of searchspace object

# This part shows how the user can create a searchspace object.
# In general, searchspaces can be continuous, discrete or hybrid.
# This depends on the parameters of the searchspace.
# In this examples, a basic discrete searchspace is presented.
# Discrete variables can be numerical, categorical or encoded chemical substances.

# To create a searchspace, we need to define all parameters that can vary between experiments.

# This example presents the optimization of a direct Arylation reaction.
# For this, we require data for solvents, ligands and bases.

# The available solvents, bases and ligands are described in the following dictionaries via SMILES.

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

# This part shows how to create the parameter objects that are used to create the campaign.
# We define the chemical substances parameters using the dictionaries defined previously.
# Here, we use `"MORDRED"` encoding, but others are available.

# We proceed to define numerical discrete parameters: `temperature` and `concentration`.

solvent = SubstanceParameter("Solvent", data=dict_solvent, encoding="MORDRED")
base = SubstanceParameter("Base", data=dict_base, encoding="MORDRED")
ligand = SubstanceParameter("Ligand", data=dict_ligand, encoding="MORDRED")

# Define numerical discrete parameters: Temperature, Concentration

temperature = NumericalDiscreteParameter(
    "Temperature", values=[90, 105, 120], tolerance=2
)
concentration = NumericalDiscreteParameter(
    "Concentration", values=[0.057, 0.1, 0.153], tolerance=0.005
)

# To simplify the creation of the campaign, we collect all parameters in a single list.

parameters = [solvent, base, ligand, temperature, concentration]

# The searchspace object can now be easily created as follows.

searchspace = SearchSpace.from_product(parameters=parameters)

#### Creation of objective object

# In this part we specify the objective of the optimization process.
# In this example, we consider a single numerical target.
# The user indicates the target variable as well as what he is trying to achieve.
# That is, the user can decide whether to maximize, minimize or match a specific value.
# In this example, we maximize the yield of a reaction.
# Hence, we indicate that the target is numerical, named `"yield"` and use the mode `"MAX"`.

objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="yield", mode="MAX")]
)

#### Creation of a campaign

# We now finally create the campaign using the objects configure previously.

campaign = Campaign(
    searchspace=searchspace,
    objective=objective,
)

# Note that an additional strategy object can be specified while creating the campaign.
# This object and its parameters are described in the basic example 'strategies'
# If no strategy is supplied, a default one is used.
# Details on strategies can be found in [`strategies`](./strategies.md)

#### Getting a recommendation

# In this part we use the campaign to recommend the next experiments to be conducted.
# To do so we use the `recommend()` function of the campaign.

# The user can specify the size of the batch of recommendations desired.
# The value needs to be an integer >= 1.

recommendation = campaign.recommend(batch_quantity=1)

print("\n\nRecommended measurements with batch_quantity = 1: ")
print(recommendation)

# `recommendation` is a dataframe with one column per parameter.
# Each row is a suggested experiment filled with a value to try for each parameter.

# If we set a greater batch quantity, the `recommendation` dataframe contains more rows.

for batch_quantity in [2, 3]:
    recommendation = campaign.recommend(batch_quantity=batch_quantity)
    print(f"\n\nRecommended measurements with batch_quantity = {batch_quantity}: ")
    print(recommendation)

#### Adding a measurement

# In this part we add target values obtained while conducting new measurements.
# This is done by creating a new column in the `recommendation` dataframe named after the target.
# In this example, we use the `add_fake_results()` utility function to create some fake results.

add_fake_results(recommendation, campaign)
print("\n\nRecommended experiments with fake measured values: ")
print(recommendation)

# The recommendation dataframe now has a new column named `yield` filled with fake values.

# Finally, we update the campaign by adding the measurement.

campaign.add_measurements(recommendation)
