"""
Run history simulation for a direct arylation where all possible combinations have
been measured
"""

from os import getcwd

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from baybe.core import BayBE
from baybe.parameters import GenericSubstance, NumericDiscrete
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.strategies.bayesian import SequentialGreedyRecommender
from baybe.strategies.sampling import RandomRecommender
from baybe.strategies.strategy import Strategy
from baybe.targets import NumericalTarget, Objective

# --------------------------------------------------------------------------------------
# PART 1: CONFIGURATION OF THE PARAMETERS AND TEST
# --------------------------------------------------------------------------------------
# Capitalized  parameters are kept meant to be fixed in this example.
# Non-capitalized are varied during this example.
# To signifiy this, we initialize everything, even if some parameters are "marked" None
# Example-specific details are listed under the NOTE below

# PART 1.1: To be adjusted by the user
batch_quantity = 1  # pylint: disable=C0103

# PART 1.2: Derived and fixed hyperparameters one can adjust
INITIAL_RECOMMENDER = RandomRecommender()
ALLOW_REPEATED_RECOMMENDATIONS = False
ALLOW_RECOMMENDING_ALREADY_MEASURED = False
NUM_MEAS_WITHIN_TOLERANCE = True
N_EXP_ITERATIONS = 10 + (batch_quantity == 1) * 20
N_MC_ITERATIONS = 5 + (batch_quantity == 1) * 3
file_name = f"AQ_Functions_IT{N_EXP_ITERATIONS}_MC{N_MC_ITERATIONS}_BQ{batch_quantity}"

# PART 1.3: Things that should not be changed resp. are contained for completeness
LOOKUP = pd.read_excel(f"{getcwd()}/examples/Reaction_DirectArylation/lookup.xlsx")
recommender = None  # pylint: disable=C0103

# NOTE Example specific details:
#   - If batch_quantity is set to 1, all available acquisition functions are tested.
#       - In this case, more Monte Carlo runs and iterations are done to get more
#           meaningful results.
#   - If a larger value is chosen, only those that allow batching, i.e., 'q'-variants
#       of the acquisition functions are tested

# --------------------------------------------------------------------------------------
# PART 2: DATA
# --------------------------------------------------------------------------------------
# In this part, we define data that is relevant for the specific example like solvents,
# ligands and so on.

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

# TODO is there a more elegant solution here?

generic_substance_dict = {
    "Solvent": (dict_solvent, "MORDRED"),
    "Base": (dict_base, "MORDRED"),
    "Ligand": (dict_ligand, "MORDRED"),
}


num_discrete_dict = {
    "Temp_C": ([90, 105, 120], 2),
    "Concentration": ([0.057, 0.1, 0.153], 0.005),
}

# --------------------------------------------------------------------------------------
# PART 3: CONSTRUCTION OF BAYBE OBJECTS
# --------------------------------------------------------------------------------------
# In this part, we collect all the data and values from the first two parts and
# construct the respective BayBE objects

numeric_discrete_parameters = [
    NumericDiscrete(name, values=val, tolerance=tol)
    for name, [val, tol] in num_discrete_dict.items()
]

generic_substance_parameters = [
    GenericSubstance(name, data=data, encoding=encoding)
    for name, [data, encoding] in generic_substance_dict.items()
]

searchspace = SearchSpace.create(
    parameters=generic_substance_parameters + numeric_discrete_parameters
)

objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="yield", mode="MAX")]
)

acq_functions = (
    ["qPI", "qEI", "qUCB"]
    if batch_quantity > 1
    else ["qPI", "qEI", "qUCB", "PM", "PI", "EI", "UCB"]
)
stratgy_dict = {}
for acq_function in acq_functions:
    stratgy_dict[acq_function] = Strategy(
        initial_recommender=INITIAL_RECOMMENDER,
        recommender=SequentialGreedyRecommender(acquisition_function_cls=acq_function),
        allow_repeated_recommendations=ALLOW_REPEATED_RECOMMENDATIONS,
        allow_recommending_already_measured=ALLOW_RECOMMENDING_ALREADY_MEASURED,
    )

# --------------------------------------------------------------------------------------
# PART 4: PUTTING THINGS TOGETHER
# --------------------------------------------------------------------------------------
# For putting things together, we define all test scenarios and simulate them.
# Afterwards, the results are plotted and saved in a file.

scenarios = {
    name: BayBE(
        searchspace=searchspace,
        strategy=strategy,
        objective=objective,
        numerical_measurements_must_be_within_tolerance=NUM_MEAS_WITHIN_TOLERANCE,
    )
    for name, strategy in stratgy_dict.items()
}

results = simulate_scenarios(
    scenarios=scenarios,
    lookup=LOOKUP,
    n_exp_iterations=N_EXP_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
    batch_quantity=batch_quantity,
)

sns.lineplot(
    data=results, x="Num_Experiments", y="yield_CumBest", hue="Variant"
).set_title(file_name)
plt.gcf().set_size_inches(22, 8)
plt.savefig(f"{getcwd()}/examples/Results/{file_name}.png")
