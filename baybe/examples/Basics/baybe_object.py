"""
Basic Example about creation and use of baybe objects
on direct arylation reaction example
"""

from baybe.core import BayBE
from baybe.parameters import GenericSubstance, NumericDiscrete
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, Objective
from baybe.utils import add_fake_results


# This example shows how to create a Baybe Object and how to use it
# It details how a user can first define parameters of the searchspace
# and the objective desired, to then be able to create a proper baybe object
# that can be used to get recommendations


# --------------------------------------------------------------------------------------
# PART 1: Creation of searchspace object
# --------------------------------------------------------------------------------------


# In this part the user define the search space
# It can be continuous, discrete or hybrid
# Here a basic discrete searchspace is presented
# Discrete variables can be numerical, categorical or encoded chemical substances

# To be able to create a searchspace in which baybe can operate
# It is important to first define all parameters that can vary between experiments
# and of course the different values that can be taken by a parameter


# Part 1.1: Define data
# --------------------------------------------------------------------------------------

# Here, we define data that is relevant for the specific example like solvents,
# ligands, temperature and so on

# This example presents the optimization of a direct Arylation reaction
# Different input parameters are varied in order to find the configuration
# that maximize the yield of the reaction
# The experimenter can vary the chemical substances used (Solvent, Base and Ligand)
# But also the temperature and the base concentration

# Each available solvents, bases and ligands are discribed in the following dictionaries
# with their corresponding SMILES

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
    "Tricyclohexylphosphine": r"P(C1CCCCC1)(C2CCCCC2)C3CCCCC3",
    "PPh3": r"P(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3",
    "XPhos": r"CC(C1=C(C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C(C(C)C)=CC(C(C)C)=C1)C",
    "P(2-furyl)3": r"P(C1=CC=CO1)(C2=CC=CO2)C3=CC=CO3",
    "Methyldiphenylphosphine": r"CP(C1=CC=CC=C1)C2=CC=CC=C2",
    "1268824-69-6": r"CC(OC1=C(P(C2CCCCC2)C3CCCCC3)C(OC(C)C)=CC=C1)C",
    "JackiePhos": r"FC(F)(F)C1=CC(P(C2=C(C3=C(C(C)C)C=C(C(C)C)C=C3C(C)C)C(OC)=CC=C2OC)"
    r"C4=CC(C(F)(F)F)=CC(C(F)(F)F)=C4)=CC(C(F)(F)F)=C1",
    "SCHEMBL15068049": r"C[C@]1(O2)O[C@](C[C@]2(C)P3C4=CC=CC=C4)(C)O[C@]3(C)C1",
    "Me2PPh": r"CP(C)C1=CC=CC=C1",
}


# Part 1.2: Define Parameters
# --------------------------------------------------------------------------------------
# Then the user define each parameter and its type
# before gathering each parameters in a single list

# Define generic chemical substances parameters: Solvent, Base and Ligand
# Here, MORDRED encoding is used for chemical substances

solvent = GenericSubstance("Solvent", data=dict_solvent, encoding="MORDRED")
base = GenericSubstance("Base", data=dict_base, encoding="MORDRED")
ligand = GenericSubstance("Ligand", data=dict_ligand, encoding="MORDRED")


# Define numerical discrete parameters: Temperature, Concentration

temperature = NumericDiscrete("Temperature", values=[90, 105, 120], tolerance=2)
concentration = NumericDiscrete(
    "Concentration", values=[0.057, 0.1, 0.153], tolerance=0.005
)

# Define the  list of parameters
parameters = [solvent, base, ligand, temperature, concentration]

# Part 1.3: Searchspace Object
# --------------------------------------------------------------------------------------

# The object searchspace can now be easily created as follows

searchspace = SearchSpace.create(parameters=parameters)


# --------------------------------------------------------------------------------------
# PART 2: Creation of objective object
# --------------------------------------------------------------------------------------


# In this part we specify the objective of the optimization process


# Here we consider a single numerical target
# The user indicates the target variable as well as the action he is trying to achieve
# It can be either maximize, minimize or match a specific value
# In this example we try to maximize the yield of a reaction
# so we indicate that the target is numerical, named 'yield'
# and that we work with the mode 'MAX'

# The Objective object is thus defined as follows

objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="yield", mode="MAX")]
)


# --------------------------------------------------------------------------------------
# PART 3: Creation of a Baybe object
# --------------------------------------------------------------------------------------


# In this part we finaly create the Baybe Object using the objects configured
# in the previous parts

baybe_obj = BayBE(
    searchspace=searchspace,
    objective=objective,
)

# NOTE: an additional object strategy can be specify while creating the baybe object
# This object and its parameters are described in the basic example 'strategies'
# If no strategy is supplied a default one is used


# --------------------------------------------------------------------------------------
# PART 4: Get a Recommendation
# --------------------------------------------------------------------------------------


# In this part we use the baybe object to recommend the next experiments to be conducted
# To do so we use the property recommend of the baybe object


# The user can specify the size of the batch of recommendations desired
# The value needs to be an integer >= 1

recommendation = baybe_obj.recommend(batch_quantity=1)

print("\n\nRecommended measurements with batch_quantity = 1: ")
print(recommendation)

# recommendation is a dataframe with columns labeled after the different variables
# Each row is a suggested experiment filled with a value to try for each parameters

# If we set a greater batch quantity,
# the recommendation dataframe would then look like this

for batch_quantity in [2, 3]:
    recommendation = baybe_obj.recommend(batch_quantity=batch_quantity)
    print(f"\n\nRecommended measurements with batch_quantity = {batch_quantity}: ")
    print(recommendation)

# --------------------------------------------------------------------------------------
# PART 5: Add a measurement
# --------------------------------------------------------------------------------------


# In this part we add target values obtained while conducting new measurements

# Part 5.1: collect target values
# --------------------------------------------------------------------------------------

# A new column is added to the recommendation dataframe named after the target variable
# The target values are inserted in this dataframe

# For the example we add fake results
add_fake_results(recommendation, baybe_obj)
print("\n\nRecommended experiments with fake measured values: ")
print(recommendation)
# The recommendation dataframe now has a new column named yield filled with fake values

# Part 5.2: Add the new measurements to the Baybe Object
# --------------------------------------------------------------------------------------

baybe_obj.add_measurements(recommendation)

# The baybe object can now be used to ask for a new recommendation
