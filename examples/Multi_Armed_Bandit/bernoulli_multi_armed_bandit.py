## Bernoulli Multi Armed Bandit

# The goal of this example is twofold:
# - It shows how to use the {class}`~baybe.surrogates.bandit.BetaBernoulliMultiArmedBanditSurrogate` for use cases like A/B testing.
# - It demonstrates the difference between **optimization** and **active learning** in this context, by employing different acquisition functions chosen for each purpose.
#   Results are compared in terms of the achieved optimality of the resulting policy and the induced system identification error.

### Imports

import os
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.stats import bernoulli

from baybe import Campaign
from baybe.acquisition import PosteriorStandardDeviation, qThompsonSampling
from baybe.acquisition.base import AcquisitionFunction
from baybe.parameters import CategoricalParameter
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.surrogates import BetaBernoulliMultiArmedBanditSurrogate
from baybe.targets import BinaryTarget
from baybe.utils.random import set_random_seed

### An Imaginary Use Case

# Bandit models are a great choice when being faced with multiple (finite) options to choose from.
# Imagine yourself running a marketing campaign where you need to decide which version of marketing material to present to your customers.
# As in all other Bayesian optimization contexts, you are confronted with the classic exploration-exploitation dilemma:
# - **Exploitation:** On the one hand, you want to choose the most effective option right from the start.
# - **Exploration:** On the other hand, you cannot know in advance how effective each option is until you have tested it.
#
# Bandit models offer a principled approach to handling this dilemma.
# They can help you determine the effect size (e.g., the click-through rate) for your marketing options or optimize the effect size
# while the campaign is still running. Both goals are achieved by dynamically selecting the version of the marketing campaign that is shown to each customer.

### The Black Box Model

# Abstractly, we can think of our problem as a black box system with a finite number of actions that can be triggered.
# The image that comes to mind is that of a slot machine with multiple arms that can be pulled (hence the name "multi-armed bandit"), each offering a different win rate.
# You can think of each arm as representing a version of the marketing campaign, and the corresponding win rates would be the true effect size of each marketing material version.

# We can build such a black box system with just a few lines of code:


@dataclass
class MultiArmedBandit:
    """A stateless black box system offering a finite number of actions (= arms)."""

    win_rates: list[float]
    """Win rates associated with the arms."""

    def draw_arm(self, arm_index: int) -> bool:
        """Draw a reward by pulling the specified arm."""
        return bernoulli(self.win_rates[arm_index]).rvs()


### Setup

# Setting a random seed for reproducibility.

set_random_seed(42)

# For our example, we use a system with a fixed collection of win rates, which we
# assume are unknown to us:

WIN_RATES = [0.3, 0.4, 0.6, 0.65, 0.8]
BANDIT = MultiArmedBandit(WIN_RATES)

# Given this target system, we consider the following two tasks:
# - **Online optimization:** Maximizing the total reward accumulated during a campaign
# - **Active learning:** Estimating the win rates of all arms

# Solving these task can be achieved through an appropriate choice of acquisition function:
# For online optimization, we use {class}`~baybe.acquisition.acqfs.qThompsonSampling`.
# For active learning, we switch to {class}`~baybe.acquisition.acqfs.PosteriorStandardDeviation`.

# To estimate the corresponding effects, we simulate each campaign for a certain number of steps and repeat this process in multiple Monte Carlo runs:

SMOKE_TEST = "SMOKE_TEST" in os.environ

ACQFS = [
    qThompsonSampling(),  # Online optimization
    PosteriorStandardDeviation(),  # Active learning
]
N_MC_RUNS = 3 if SMOKE_TEST else 10
N_ITERATIONS = 50 if SMOKE_TEST else 200


### Building the Model

# To model the rewards obtained from the bandit, we use a {class}`~baybe.targets.binary.BinaryTarget`.
# The corresponding search space is spanned by a {class}`~baybe.parameters.categorical.CategoricalParameter` describing the bandit's arms.

n_arms = len(WIN_RATES)
target = BinaryTarget(name="clicked")
parameter = CategoricalParameter(
    name="Bandit Arm",
    values=[str(i) for i in range(n_arms)],
    # For bandits, one-hot parameter encoding is required:
    encoding="OHE",
)
surrogate = BetaBernoulliMultiArmedBanditSurrogate()


# For each simulation, we report the trajectory of earned rewards and the estimated win rates of the bandit arms:


class SimulationResult(NamedTuple):
    """Result of a bandit campaign simulation."""

    earned_rewards: np.ndarray
    estimated_win_rates: np.ndarray


# Putting all pieces together, we define a function to simulate our campaign under a given acquisition strategy:


def simulate(acqf: AcquisitionFunction) -> SimulationResult:
    """Simulate the campaign with with the given acquisition function."""
    recommender = TwoPhaseMetaRecommender(
        initial_recommender=RandomRecommender(),
        recommender=BotorchRecommender(
            surrogate_model=surrogate,
            acquisition_function=acqf,
            # The same arm can be pulled several times:
            allow_repeated_recommendations=True,
            allow_recommending_already_measured=True,
        ),
    )

    # Containers for storing the rewards and win rate estimates
    earned_rewards = np.zeros((N_MC_RUNS, N_ITERATIONS))
    estimated_win_rates = np.zeros((N_MC_RUNS, n_arms))

    for mc in range(N_MC_RUNS):
        searchspace = parameter.to_searchspace()
        objective = target.to_objective()
        campaign = Campaign(searchspace, objective, recommender)

        for i in range(N_ITERATIONS):
            df = campaign.recommend(batch_size=1)
            recommended_arm_idx = int(df.values.item())
            reward = BANDIT.draw_arm(recommended_arm_idx)
            earned_rewards[mc, i] = reward
            df["clicked"] = reward
            campaign.add_measurements(df)

        estimated_win_rates[mc] = surrogate.posterior_means()

    return SimulationResult(earned_rewards, estimated_win_rates)


### Running the Simulation

# With our model in place, we can now simulate the campaign for both acquisition functions.
# The success of our two objectives is quantified using the following metrics:
# - For online optimization, we look at the **empirically achieved win rate** and
#   compare it to the win rate of the best arm.
# - For active learning, we report the **mean squared error of the win rates**
#   estimated at the end of the campaign.

for acqf in ACQFS:
    name = acqf.__class__.__name__
    result = simulate(acqf)
    estimates = result.estimated_win_rates
    estimation_bias = estimates.mean(axis=0) - WIN_RATES
    estimation_variance = estimates.var(axis=0)
    MSE = np.sum(estimation_bias**2 + estimation_variance)

    print(name)
    print("-" * len(name))
    print("Optimal win rate:", max(WIN_RATES))
    print("Achieved win rate:", result.earned_rewards.mean())
    print("Mean squared error (last iteration):", MSE)
    print("\n")


### Discussion

# As expected, we observe a higher achieved win rate for the online optimization strategy, as its underlying acquisition function is designed to recommend the arm with the largest reward most often.
# This comes at the cost of a higher estimation error for the remaining arms, which are sampled less frequently.
# In contrast, the active learning strategy attempts to balance the estimation errors across arms, resulting in a significantly smaller overall error.
# However, in order to do so, it must sample arms with low and high win rates equally often, which reduces the achieved rate and thus the total reward accumulated in the campaign.
