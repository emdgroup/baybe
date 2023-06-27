"""
Description of the different strategies available.

# This example explain how to create the strategy object

This object specifies the strategy adopted to make recommendations
It has several parameters one can adjust
depending on the strategy the user wants to follow
To apply the selected strategy,
this object can be specified in the arguments of the baybe object

The different parameters the user can change are:
- The initial recommender
- The recommender with its surrogate model and its acquisition function
- Other parameters to allow or not repetition of recommendations
This example explain how they can be changed
"""


from baybe.core import BayBE
from baybe.parameters import GenericSubstance, NumericDiscrete
from baybe.searchspace import SearchSpace
from baybe.strategies.bayesian import SequentialGreedyRecommender
from baybe.strategies.sampling import RandomRecommender
from baybe.strategies.strategy import Strategy
from baybe.targets import NumericalTarget, Objective
from baybe.utils import add_fake_results


# --------------------------------------------------------------------------------------
# PART 1: Initial Strategy
# --------------------------------------------------------------------------------------


# This part describes how one can choose the initial strategy to start the optimization

# For the first recommendation, the user can specify which strategy to adopt
# All available recommenders are listed below
# One should not forget to make the corresponding import before using them

initial_recommenders = [
    "Random",  #: RandomRecommender(),
    "Farthest Point Sampling",  # FPSRecommender(),
    "KMEANS Clustering",  # KMeansClusteringRecommender(),
]


# Per default the initial recommender chosen is a random recommender
INITIAL_RECOMMENDER = RandomRecommender()

# --------------------------------------------------------------------------------------
# PART 2: Surrogate Model
# --------------------------------------------------------------------------------------

# This part describes the different surrogate models that can be used

# This model uses available data to model the objective function as well as the
# uncertainty of the model.
# The model is then used by the acquisition function to make recommendations

# All available surrogate models are listed below:
available_surrogate_models = [
    "GP",  # GaussianProcessModel
    "RF",  # Random Forest Model
    "NG",  # Natural Gradient Boosting
    "BL",  # Bayesian Linear Regression
]

# Per default a Gaussian Process is used
SURROGATE_MODEL = "GP"


# STRATEGY OBJECT
# --------------------------------------------------------------------------------------

# In the strategy object a recommender object can be specified
# In a bayesian approach this recommender object should be a SequentialGreedyRecommender
# This Recommender uses a surrogate model to model the data
# and then an acquisition function to select candidates for new recommendations
# The chosen surrogate model can be specified as argument of SequentialGreedyRecommender
# strategy = Strategy(
# recommender=SequentialGreedyRecommender(surrogate_model_cls=SURROGATE_MODEL)


# --------------------------------------------------------------------------------------
# PART 3: Acquisition function
# --------------------------------------------------------------------------------------

# In this part we describe the different acquisition function that can be employed

# This function looks for points where measurements of the target value
# could effectively improve the model

# All available acquisition functions are listed below:

available_acq_functions = [
    "qPI",  # q-Probability Of Improvement
    "qEI",  # q-Expected Improvement
    "qUCB",  # q-upper confidence bound with beta of 1.0
    "PM",  # Posterior Mean,
    "PI",  # Probability Of Improvement,
    "EI",  # Expected Improvement,
    "UCB",  # upper confidence bound with beta of 1.0
]

# NOTE Example specific details:
#   - If batch_quantity is set to 1, all available acquisition functions can be chosen
#
#   - If a larger value is chosen, only those that allow batching, i.e., 'q'-variants
#       of the acquisition functions must be chosen

# Per default the acquisition function is qExpected Improvement
ACQ_FUNCTION = "qEI"

# STRATEGY OBJECT
# --------------------------------------------------------------------------------------

# Similarly as with surrogate models, the acquisition fucntion is defined
# in the recommender object
# strategy = Strategy(
#    recommender=SequentialGreedyRecommender(acquisition_function_cls=ACQ_FUNCTION))


# --------------------------------------------------------------------------------------
# PART 3: Other parameters
# --------------------------------------------------------------------------------------


# Two other boolean hyperparameters can be specified while creating a strategy object

# One to allow the recommendation of points that are already recommended
# One to allow the recommendation of points that have already been measured

# per default, they are set to True
ALLOW_REPEATED_RECOMMENDATIONS = True
ALLOW_RECOMMENDING_ALREADY_MEASURED = True

# STRATEGY OBJECT
# --------------------------------------------------------------------------------------

# Like the initial recommender these parameters can be specified in the arguments
# of the strategy object
# strategy = Strategy(
#    allow_repeated_recommendations=ALLOW_REPEATED_RECOMMENDATIONS,
#    allow_recommending_already_measured=ALLOW_RECOMMENDING_ALREADY_MEASURED)


# --------------------------------------------------------------------------------------
# PART 4: Create the Strategy Object
# --------------------------------------------------------------------------------------


# To create the strategy object each parameter described above can be specified as
# follows, Please note that they all have default values and therefore one does not need
# to specify all of them to create a strategy object

strategy = Strategy(
    initial_recommender=INITIAL_RECOMMENDER,
    recommender=SequentialGreedyRecommender(
        surrogate_model_cls=SURROGATE_MODEL, acquisition_function_cls=ACQ_FUNCTION
    ),
    allow_repeated_recommendations=ALLOW_REPEATED_RECOMMENDATIONS,
    allow_recommending_already_measured=ALLOW_RECOMMENDING_ALREADY_MEASURED,
)

print(strategy)


# --------------------------------------------------------------------------------------
# PART 5: Incorporate chosen strategy in BayBE object
# --------------------------------------------------------------------------------------


# In order to adopt the chosen strategy in the optimization process
# The strategy object need to be specified while creating the BayBE object

# As seen in the baybe_object basic example to do so one needs
# to create first searchspace and objective objects
# then together with the strategy object they can be used to create a BayBE object

# Part 5.1: Example Searchspace and objective parameters
# --------------------------------------------------------------------------------------

# For this example, the data from the baybe_object example are used
# e.g. DirectArylation data
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

# Define list of parameters

solvent = GenericSubstance("Solvent", data=dict_solvent, encoding="MORDRED")
base = GenericSubstance("Base", data=dict_base, encoding="MORDRED")
ligand = GenericSubstance("Ligand", data=dict_ligand, encoding="MORDRED")
temperature = NumericDiscrete("Temperature", values=[90, 105, 120], tolerance=2)
concentration = NumericDiscrete(
    "Concentration", values=[0.057, 0.1, 0.153], tolerance=0.005
)

parameters = [solvent, base, ligand, temperature, concentration]

# Creation of searchspace and objective Objects

searchspace = SearchSpace.create(parameters=parameters)

objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="yield", mode="MAX")]
)


# Part 5.2: Creation of the BayBE Object
# --------------------------------------------------------------------------------------

# During creation of a BayBE object one can specify the strategy adopted

baybe_obj = BayBE(
    searchspace=searchspace,
    strategy=strategy,
    objective=objective,
)

# This baybe object can then be used to get recommendations and add measurements

recommendation = baybe_obj.recommend(batch_quantity=3)
print("\n\nRecommended experiments: ")
print(recommendation)

add_fake_results(recommendation, baybe_obj)
print("\n\nRecommended experiments with fake measured values: ")
print(recommendation)

baybe_obj.add_measurements(recommendation)
