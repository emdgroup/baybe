### Example for full simulation loop using a table-based lookup mechanism with incomplete data

"""
This example shows a simulation for a direct arylation where not all combinations have been measured
This allows us to access information about previously conducted experiments from .xlsx-files.
"""

# This examples assumes some basic familiarity with using BayBE and the lookup mechanism.
# We refer to [`baybe_object`](./../Basics/baybe_object.md) for a more  basic example resp.
# to [`run_full_lookup`](./run_full_lookup.md) for details on the lookup mechanism.

#### Necessary imports for this example

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from baybe.core import BayBE
from baybe.parameters import GenericSubstance, NumericDiscrete
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.strategies.sampling import RandomRecommender
from baybe.strategies.strategy import Strategy
from baybe.targets import NumericalTarget, Objective

### Parameters for a full simulation loop

# For the full simulation, we need to define some additional parameters.
# These are the number of Monte Carlo runs and the number of experiments to be conducted per run.
N_MC_ITERATIONS = 2
N_EXP_ITERATIONS = 5

### Lookup functionality and data creation

# See [`run_full_lookup`](./run_full_lookup.md) for details.
try:
    lookup = pd.read_excel("./lookup_withmissing.xlsx")
except FileNotFoundError:
    try:
        lookup = pd.read_excel("examples/Backtesting/lookup_withmissing.xlsx")
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

solvent = GenericSubstance(name="Solvent", data=dict_solvent, encoding="MORDRED")
base = GenericSubstance(name="Base", data=dict_base, encoding="MORDRED")
ligand = GenericSubstance(name="Ligand", data=dict_ligand, encoding="MORDRED")
temperature = NumericDiscrete(name="Temp_C", values=[90, 105, 120], tolerance=2)
concentration = NumericDiscrete(
    name="Concentration", values=[0.057, 0.1, 0.153], tolerance=0.005
)

parameters = [solvent, base, ligand, temperature, concentration]

searchspace = SearchSpace.from_product(parameters=parameters)
objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="yield", mode="MAX")]
)

### Constructing BayBE objects for the simulation loop

# In this example, we create two BayBE objects.
# One uses the default recommender and the other one makes random recommendations.

baybe = BayBE(searchspace=searchspace, objective=objective)
baybe_rand = BayBE(
    searchspace=searchspace,
    strategy=Strategy(recommender=RandomRecommender()),
    objective=objective,
)

# We can now use the `simulate_scenarios` function to simulate a full experiment.
# Note that this function enables to run multiple scenarios by a single function call.
# For this, it is necessary to define a dictionary mapping scenario names to BayBE objects.
scenarios = {"Test_Scenario": baybe, "Random": baybe_rand}

# The lookup table does not contain data for all possible combination of parameters.
# Consequently, we need to inform the function how to deal with missing entries.
# This is done via the `impute_mode` keyword.
# The following options are available:
#   * `"error"`: an error will be thrown
#   * `"worst"`: imputation using the worst available value for each target
#   * `"best"`: imputation using the best available value for each target
#   * `"mean"`: imputation using mean value for each target
#   * `"random"`: a random row will be used as lookup
#   * `"ignore"`: the search space is stripped before recommendations are made
#       so that unmeasured experiments will not be recommended

results = simulate_scenarios(
    scenarios=scenarios,
    batch_quantity=3,
    n_exp_iterations=N_EXP_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
    lookup=lookup,
    impute_mode="best",
)

# The following lines plot the results and save the plot in run_impute_mode.png
max_yield = lookup["yield"].max()
sns.lineplot(
    data=results, x="Num_Experiments", y="yield_CumBest", hue="Variant", marker="x"
)
plt.plot([3, 3 * N_EXP_ITERATIONS], [max_yield, max_yield], "--r")
plt.legend(loc="lower right")
plt.gcf().set_size_inches(20, 8)
plt.savefig("./run_impute_mode.png")
