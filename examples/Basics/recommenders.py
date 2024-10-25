## Example for using different strategies

# This example shows how to create and use recommender objects.
# Such an object specifies the recommender adopted to make recommendations.
# It has several parameters one can adjust, depending on the recommender the user wants to follow.

# To apply the selected recommender, this object can be specified in the arguments of the campaign.
# The different parameters the user can change are:
# - The initial recommender
# - The recommender with its surrogate model and its acquisition function
# - Other parameters to allow or not repetition of recommendations


# This examples assumes some basic familiarity with using BayBE.
# We refer to [`campaign`](./campaign.md) for a more general and basic example.

### Necessary imports for this example

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, SubstanceParameter
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.base import Surrogate
from baybe.targets import NumericalTarget
from baybe.utils.basic import get_subclasses
from baybe.utils.dataframe import add_fake_measurements

### Available recommenders suitable for initial recommendation

# For the first recommendation, the user can specify which recommender to use.
# The following initial recommenders are available.
# Note that it is necessary to make the corresponding import before using them.

initial_recommenders = [
    "Random",  #: RandomRecommender(),
    "Farthest Point Sampling",  # FPSRecommender(),
    "KMEANS Clustering",  # KMeansClusteringRecommender(),
]


# Per default the initial recommender chosen is a random recommender.

INITIAL_RECOMMENDER = RandomRecommender()

### Available surrogate models

# This model uses available data to model the objective function as well as the uncertainty.
# The surrogate model is then used by the acquisition function to make recommendations.

# The following are the available basic surrogates:

for subclass in get_subclasses(Surrogate):
    print(subclass)

# Per default a Gaussian Process is used
# You can change the used kernel by using the optional `kernel` keyword.

SURROGATE_MODEL = GaussianProcessSurrogate()


### Acquisition function

# This function looks for points where measurements of the target value could improve the model.
# The following acquisition functions are generally available.

available_acq_functions = [
    "qPI",  # q-Probability Of Improvement
    "qEI",  # q-Expected Improvement
    "qUCB",  # q-upper confidence bound with beta of 1.0
    "PM",  # Posterior Mean,
    "PI",  # Probability Of Improvement,
    "EI",  # Expected Improvement,
    "UCB",  # upper confidence bound with beta of 1.0
]

# Note that the qvailability of the acquisition functions might depend on the `batch_size`:
#   - If `batch_size` is set to 1, all available acquisition functions can be chosen
#   - If a larger value is chosen, only those that allow batching.
#       That is, 'q'-variants of the acquisition functions must be chosen.

# The default he acquisition function is q-Expected Improvement.

ACQ_FUNCTION = "qEI"

### Other parameters

# Two other boolean hyperparameters can be specified when creating a recommender object.
# The first one allows the recommendation of points that were already recommended previously.
# The second one allows the recommendation of points that have already been measured.
# Per default, they are set to `True`.

ALLOW_REPEATED_RECOMMENDATIONS = True
ALLOW_RECOMMENDING_ALREADY_MEASURED = True

### Creating the recommender object

# To create the recommender object, each parameter described above can be specified as follows.
# Note that they all have default values.
# Therefore one does not need to specify all of them to create a recommender object.

recommender = TwoPhaseMetaRecommender(
    initial_recommender=INITIAL_RECOMMENDER,
    recommender=BotorchRecommender(
        surrogate_model=SURROGATE_MODEL,
        acquisition_function=ACQ_FUNCTION,
        allow_repeated_recommendations=ALLOW_REPEATED_RECOMMENDATIONS,
        allow_recommending_already_measured=ALLOW_RECOMMENDING_ALREADY_MEASURED,
    ),
)

print(recommender)

# Note that there are the additional keywords `hybrid_sampler` and `sampling_percentag`.
# Their meaning and how to use and define it are explained in the hybrid backtesting example.
# We thus refer to [`hybrid`](./../Backtesting/hybrid.md) for details on these.

### Example Searchspace and objective parameters

# We use the same data used in the [`campaign`](./campaign.md) example.

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

solvent = SubstanceParameter("Solvent", data=dict_solvent, encoding="MORDRED")
base = SubstanceParameter("Base", data=dict_base, encoding="MORDRED")
ligand = SubstanceParameter("Ligand", data=dict_ligand, encoding="MORDRED")
temperature = NumericalDiscreteParameter(
    "Temperature", values=[90, 105, 120], tolerance=2
)
concentration = NumericalDiscreteParameter(
    "Concentration", values=[0.057, 0.1, 0.153], tolerance=0.005
)

# We collect all parameters in a list.

parameters = [solvent, base, ligand, temperature, concentration]

# We create the searchspace and the objective.

searchspace = SearchSpace.from_product(parameters=parameters)

objective = SingleTargetObjective(target=NumericalTarget(name="yield", mode="MAX"))

### Creating the campaign

# The recommender object can now be used together with the searchspace and the objective as follows.

campaign = Campaign(
    searchspace=searchspace,
    recommender=recommender,
    objective=objective,
)

# This campaign can then be used to get recommendations and add measurements:

recommendation = campaign.recommend(batch_size=3)
print("\n\nRecommended experiments: ")
print(recommendation)

add_fake_measurements(recommendation, campaign.targets)
print("\n\nRecommended experiments with fake measured values: ")
print(recommendation)

campaign.add_measurements(recommendation)
