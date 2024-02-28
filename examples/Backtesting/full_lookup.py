## Example for full simulation loop using a table-based lookup mechanism

# This example shows a simulation for a direct arylation where all combinations have been measured.
# This allows us to access information about previously conducted experiments from .xlsx-files.

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.

### Necessary imports for this example

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from baybe import Campaign
from baybe.objective import Objective
from baybe.parameters import NumericalDiscreteParameter, SubstanceParameter
from baybe.recommenders import RandomRecommender, TwoPhaseMetaRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget

### Parameters for a full simulation loop

# For the full simulation, we need to define some additional parameters.
# These are the number of Monte Carlo runs and the number of experiments to be conducted per run.

SMOKE_TEST = "SMOKE_TEST" in os.environ

N_DOE_ITERATIONS = 2 if SMOKE_TEST else 5
N_MC_ITERATIONS = 2 if SMOKE_TEST else 3
BATCH_SIZE = 1 if SMOKE_TEST else 3

### Lookup functionality and data creation

# We read the information about the conducted experiments from a .xlsx-file.
# This data set was obtained from [Shields, B.J., Stevens et al. Nature 590, 89–96 (2021)](https://doi.org/10.1038/s41586-021-03213-y) and contains measurements of a reaction yield,
# varying typical reaction conditions.
# Depending on your system and settings, you might need to slightly adjust the following paths.
# The reason is that it depends on the folder in which you execute the `python` call.
# This code assumes that you call `python` either from the repository root folder or this folder.

try:
    lookup = pd.read_excel("./lookup.xlsx")
except FileNotFoundError:
    try:
        lookup = pd.read_excel("examples/Backtesting/lookup.xlsx")
    except FileNotFoundError as e:
        print(e)

# As usual, we set up some experiment.
# Note that we now need to ensure that the names fit the names in the provided .xlsx file!

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

### Creating the searchspace and the objective

# Here, we create the parameter objects, the searchspace and the objective.

solvent = SubstanceParameter(name="Solvent", data=dict_solvent, encoding="MORDRED")
base = SubstanceParameter(name="Base", data=dict_base, encoding="MORDRED")
ligand = SubstanceParameter(name="Ligand", data=dict_ligand, encoding="MORDRED")
temperature = NumericalDiscreteParameter(
    name="Temp_C", values=[90, 105, 120], tolerance=2
)
concentration = NumericalDiscreteParameter(
    name="Concentration", values=[0.057, 0.1, 0.153], tolerance=0.005
)

parameters = [solvent, base, ligand, temperature, concentration]

searchspace = SearchSpace.from_product(parameters=parameters)
objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="yield", mode="MAX")]
)

### Constructing campaigns for the simulation loop

# In this example, we create two campaigns.
# One uses the default recommender and the other one makes random recommendations.

campaign = Campaign(searchspace=searchspace, objective=objective)
campaign_rand = Campaign(
    searchspace=searchspace,
    recommender=TwoPhaseMetaRecommender(recommender=RandomRecommender()),
    objective=objective,
)

# We can now use the `simulate_scenarios` function to simulate a full experiment.
# Note that this function enables to run multiple scenarios by a single function call.
# For this, it is necessary to define a dictionary mapping scenario names to campaigns.

scenarios = {"Test_Scenario": campaign, "Random": campaign_rand}

results = simulate_scenarios(
    scenarios,
    lookup,
    batch_size=BATCH_SIZE,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)

# The following lines plot the results and save the plot in run_full_lookup.png

max_yield = lookup["yield"].max()
sns.lineplot(
    data=results, x="Num_Experiments", y="yield_CumBest", hue="Scenario", marker="x"
)
plt.plot([3, 3 * N_DOE_ITERATIONS], [max_yield, max_yield], "--r")
plt.legend(loc="lower right")
plt.gcf().set_size_inches(20, 8)
plt.savefig("./run_full_lookup.png")
