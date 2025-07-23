import warnings

import numpy as np
import pandas as pd

from baybe import Campaign
from baybe.exceptions import InputDataTypeWarning
from baybe.objectives import DesirabilityObjective
from baybe.parameters import CategoricalParameter, NumericalDiscreteParameter
from baybe.recommenders import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.random import set_random_seed

warnings.simplefilter("ignore", InputDataTypeWarning)

set_random_seed(12)
n = 12

# Parameter options
bases = ["K2CO3", "NaOH", "K3PO4"]
gases = ["N2", "Ar", "Kr"]
catalysts = ["APhos", "BPhos", "CPhos"]
temperatures = [40, 70]

# Sample values randomly but reproducibly
data = {
    "base": np.random.choice(bases, n),
    "gas": np.random.choice(gases, n),
    "catalyst": np.random.choice(catalysts, n),
    "temperature": np.random.choice(temperatures, n),
    "Yield": np.round(np.random.uniform(45, 95, n), 2),
    "conversion": np.round(np.random.uniform(50, 100, n), 2),
}

df = pd.DataFrame(data)

# BayBE Setup
parameters = [
    CategoricalParameter(name="base", values=bases, encoding="OHE"),
    CategoricalParameter(name="gas", values=gases, encoding="OHE"),
    CategoricalParameter(name="catalyst", values=catalysts, encoding="OHE"),
    NumericalDiscreteParameter(name="temperature", values=temperatures),
]
searchspace = SearchSpace.from_product(parameters)
recommender = BotorchRecommender()
targets = [
    NumericalTarget(name="Yield", mode="MAX", bounds=(0, 100), transformation="LINEAR"),
    NumericalTarget(
        name="conversion", mode="MAX", bounds=(0, 100), transformation="LINEAR"
    ),
]
objective = DesirabilityObjective(targets=targets, scalarizer="MEAN")
campaign = Campaign(
    searchspace=searchspace,
    objective=objective,
    recommender=recommender,
    allow_recommending_pending_experiments=True,
    allow_recommending_already_recommended=True,
)

campaign.add_measurements(df)

# experiments = campaign.recommend(batch_size=3)
# joint_acqf_value = campaign.joint_acquisition_value(candidates=experiments)
# print(joint_acqf_value)

best_joint_acqf_value = -np.inf
proposed_experiments = None
import copy

from baybe.constraints import DiscreteExcludeConstraint, SubSelectionCondition

for i, gas in enumerate(gases):
    for j, temperature in enumerate(temperatures):
        campaign_tmp = copy.deepcopy(campaign)

        gas_excluded = [x for x in gases if x != gas]
        temp_excluded = [x for x in temperatures if x != temperature]

        constraint = DiscreteExcludeConstraint(
            parameters=["gas", "temperature"],
            combiner="OR",
            conditions=[
                SubSelectionCondition(selection=gas_excluded),
                SubSelectionCondition(selection=temp_excluded),
            ],
        )

        campaign_tmp.toggle_discrete_candidates(
            constraints=[constraint], exclude=True, complement=True
        )
        experiments = campaign_tmp.recommend(batch_size=3)

        joint_acqf_value = campaign_tmp.joint_acquisition_value(candidates=experiments)

        print(joint_acqf_value, best_joint_acqf_value)

        if joint_acqf_value > best_joint_acqf_value:
            best_joint_acqf_value = joint_acqf_value
            proposed_experiments = experiments

        del campaign_tmp
