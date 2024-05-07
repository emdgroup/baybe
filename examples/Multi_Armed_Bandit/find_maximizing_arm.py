## Example for a Multi Armed Bandit

# This example shows how to use the bernoulli multi armed bandit surrogate.

from collections.abc import Iterable
from typing import Union

import numpy as np
from attrs import define
from scipy.stats import bernoulli, rv_continuous, rv_discrete

from baybe import Campaign
from baybe.acquisition import ProbabilityOfImprovement
from baybe.objectives import SingleTargetObjective
from baybe.parameters import CategoricalParameter
from baybe.recommenders import (
    FPSRecommender,
    SequentialGreedyRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.surrogates import BernoulliMultiArmedBanditSurrogate
from baybe.targets import BinaryTarget

### Setup

# We are using a 5-armed bandit in this example. The bandit has a random win rate for now.

N_ARMS = 5
N_ITERATIONS = 300
np.random.seed(0)


@define
class MultiArmedBanditModel:
    """Representation of a multi armed bandit."""

    real_distributions: list[Union[rv_discrete, rv_continuous]]
    """List of the reward distribution per arm."""

    def sample(self, arm_idxs: Iterable[int]):
        """Draw reward samples from the arms indexed in arm_idxs."""
        return [self.real_distributions[arm_idx].rvs() for arm_idx in arm_idxs]

    @property
    def means(self):
        """Return the real means of the reward distributions."""
        return [dist.stats(moments="m") for dist in self.real_distributions]


mab = MultiArmedBanditModel(
    real_distributions=[bernoulli(np.random.rand()) for _ in range(N_ARMS)]
)
print("real means", mab.means)


### Campaign

# We are using the BinaryTarget as we are modeling a bernoulli reward.
# The searchspace has one categorical parameter to model the arms of the bandit.
# The probability of improvement acquisition function is not perfect in this setting
# as it assumes a normal distribution of the win rate.

target = BinaryTarget(name="win_rate")
objective = SingleTargetObjective(target=target)
parameters = [
    CategoricalParameter(
        name="arm",
        values=[str(i) for i in range(N_ARMS)],
    )
]
searchspace = SearchSpace.from_product(parameters)
mabs = BernoulliMultiArmedBanditSurrogate()
recommender = TwoPhaseMetaRecommender(
    initial_recommender=FPSRecommender(
        allow_repeated_recommendations=True,
        allow_recommending_already_measured=True,
    ),
    recommender=SequentialGreedyRecommender(
        surrogate_model=mabs,
        allow_repeated_recommendations=True,
        allow_recommending_already_measured=True,
        acquisition_function=ProbabilityOfImprovement(),
    ),
)
campaign = Campaign(searchspace, objective, recommender)


### Optimization Loop

total_reward = 0
for i in range(N_ITERATIONS):
    df = campaign.recommend(batch_size=1)
    reward = mab.sample(df.index.tolist())
    total_reward += sum(reward)
    df["win_rate"] = reward
    campaign.add_measurements(df)

    if (i + 1) % 50 == 0:
        print("iter", i + 1)
        print("estimated means", mabs.means)
        print("-" * 5)

real_means = mab.means
print("real means", real_means)
print("optimal expected reward", max(real_means) * N_ITERATIONS)
print("total_reward", total_reward)
print("mean reward", total_reward / N_ITERATIONS)
