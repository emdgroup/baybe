## Bernoulli Multi Armed Bandit

# This example shows how to use the Bernoulli multi armed bandit surrogate for use cases like A/B Testing.
# Imagin you have 5 versions of something (e.g., a marketing campaign) that you would like to test against each other.
# The Multi-Armed Banit can help you to identify the effect size (e.g., click-through rate) or optimize the effect size
# while the experiment is running. This is done by recommending a version of the marketing campaign to show to each new user.

import os

import numpy as np
from multi_armed_bandit_model import MultiArmedBanditModel
from scipy.stats import bernoulli, beta
from torch import Tensor
from utils import create_mab_plot, max_rv_distribution

from baybe import Campaign
from baybe.acquisition import PosteriorStandardDeviation, ThompsonSampling
from baybe.objectives import SingleTargetObjective
from baybe.parameters import CategoricalParameter
from baybe.recommenders import (
    BotorchRecommender,
    FPSRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.surrogates import BernoulliMultiArmedBanditSurrogate
from baybe.targets import BinaryTarget

### Setup

# We are using a 5-armed bandit in this example. The bandit arms have a hard-coded win rate for now.
# You can think of each arm representing one version of the marketing campaign, and the win rates would be
# the true effect size of each version.
# As acquisition functions, we are using ThompsonSampling for the online optimization. For the active learning
# task, we are using the PosteriorStandardDeviation acquisition function.

N_ARMS = 5
SMOKE_TEST = "SMOKE_TEST" in os.environ
RENDER = not SMOKE_TEST
N_MC_RUNS = 10 if not SMOKE_TEST else 3
N_ITERATIONS = 300 if not SMOKE_TEST else 50
acqfs = [
    # Online optimization (maximizing the reward)
    ThompsonSampling(),
    # Active learning (learning the effect size per arm)
    PosteriorStandardDeviation(),
]
real_means = Tensor([0.3, 0.4, 0.6, 0.65, 0.8])
mab = MultiArmedBanditModel(
    real_distributions=[bernoulli(real_mean) for real_mean in real_means]
)
print("real means", real_means)

### Campaign

# We are using the BinaryTarget as we are modeling a Bernoulli reward (click or no click).
# The search space has one categorical parameter to model the arms of the bandit.

# To test the approach we are running multiple Monte Carlo runs per acquisition function.

for acqf in acqfs:
    np.random.seed(0)

    mc_estimated_means_over_iterations = []
    mc_regret_over_iterations = []
    mc_rewards = []

    # only taking posterior params for visualization of the last run
    posterior_params = None

    for mc_run_i in range(N_MC_RUNS):
        target = BinaryTarget(name="clicked")
        objective = SingleTargetObjective(target=target)
        parameters = [
            CategoricalParameter(
                name="Bandit Arms",
                values=[f"arm_{i}" for i in range(N_ARMS)],
            )
        ]
        searchspace = SearchSpace.from_product(parameters)
        surrogate = BernoulliMultiArmedBanditSurrogate()
        recommender = TwoPhaseMetaRecommender(
            initial_recommender=FPSRecommender(
                allow_repeated_recommendations=True,
                allow_recommending_already_measured=True,
            ),
            recommender=BotorchRecommender(
                surrogate_model=surrogate,
                allow_repeated_recommendations=True,
                allow_recommending_already_measured=True,
                acquisition_function=acqf,
            ),
        )
        campaign = Campaign(searchspace, objective, recommender)

        total_reward = 0
        estimated_means = []
        posterior_params = []
        regrets = []
        for i in range(N_ITERATIONS):
            df = campaign.recommend(batch_size=1)

            reward = mab.sample(df.index.tolist())
            total_reward += sum(reward)
            df["clicked"] = reward
            campaign.add_measurements(df)

            if i > 0:
                regrets.append((i + 1) * max(real_means) - total_reward)
                posterior_means = surrogate.posterior_mean_per_arm().tolist()
                estimated_means.append(posterior_means)
                posterior_params.append(surrogate._posterior_beta_parameters())

        mc_estimated_means_over_iterations.append(estimated_means)
        mc_regret_over_iterations.append(regrets)
        mc_rewards.append(total_reward)

    mc_estimated_means_over_iterations = Tensor(mc_estimated_means_over_iterations)
    mc_regret_over_iterations = Tensor(mc_regret_over_iterations)
    mc_regret_mean = mc_regret_over_iterations.mean(dim=0)
    mc_regret_std = mc_regret_over_iterations.std(dim=0)
    mc_estimated_means_squared_error = (
        (mc_estimated_means_over_iterations - real_means).pow(2).sum(dim=-1)
    )
    mc_estimated_means_squared_error_mean = mc_estimated_means_squared_error.mean(dim=0)
    mc_estimated_means_squared_error_std = mc_estimated_means_squared_error.std(dim=0)

    max_rv_distribution_over_iterations = np.array(
        [
            max_rv_distribution(posterior_param.T, beta)
            for posterior_param in posterior_params
        ]
    )

    print("-" * 10)
    print(f"Ran for {N_MC_RUNS} MC runs with {acqf.__class__.__name__}.")
    print("Optimal expected reward", (max(real_means) * N_ITERATIONS).item())
    print("Mean reward", sum(mc_rewards) / N_MC_RUNS)
    print("Mean regret last iteration", mc_regret_mean[-1].item())
    print("MSE last iteration", mc_estimated_means_squared_error_mean[-1].item())

    if RENDER:
        create_mab_plot(
            N_ITERATIONS,
            N_ARMS,
            mc_regret_mean,
            mc_regret_std,
            mc_estimated_means_squared_error_mean,
            mc_estimated_means_squared_error_std,
            max_rv_distribution_over_iterations,
            posterior_params,
            real_means,
            estimated_means,
            acqf,
            "bernoulli_multi_armed_bandit",
        )

### Outcome

# We observe a smaller regret when using the online optimization approach (Thompson sampling), which makes sense as we
# are trying to recommend the arm with the largest reward more often. The MSE, on the other hand, is larger than in the
# active learning approach. As the active learning approach will sample the next action based on the maximum standard
# deviation, we are concentrating more on learning the effect size than maximizing the reward. Despite those differences,
# both approaches identify the best arm rather quickly. This can be seen by the $P(X_i > X_{-i})$ plot stating the probability
# for each arm to generate the maximum reward.
