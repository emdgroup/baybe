## Bernoulli Multi Armed Bandit

# This example shows how to use the bernoulli multi armed bandit surrogate.

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from multi_armed_bandit_model import MultiArmedBanditModel
from scipy.stats import bernoulli, beta
from torch import Tensor
from utils import max_rv_distribution

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

# We are using a 5-armed bandit in this example. The bandit has a hard coded win rate for now.
# As acquisition functions we are taking ThompsonSampling for the online optimization task and
# the identification of the maximizing arm. For the active learning task we are taking the PosteriorStandardDeviation.

N_ARMS = 5
SMOKE_TEST = "SMOKE_TEST" in os.environ
RENDER = not SMOKE_TEST
N_MC_RUNS = 10 if not SMOKE_TEST else 3
COLORS = ["blue", "red", "green", "black", "pink"]
N_ITERATIONS = 300 if not SMOKE_TEST else 50
acqfs = [
    # Aptimization (best arm identification)
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

# We are using the BinaryTarget as we are modeling a bernoulli reward.
# The searchspace has one categorical parameter to model the arms of the bandit.

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
        fig, [top_left_ax, middle_ax, bottom_ax] = plt.subplots(
            3,
            height_ratios=(0.25, 0.25, 0.5),
        )
        fig.set_figheight(7)
        top_right_ax = top_left_ax.twinx()

        def animate(i):
            """Animate one frame."""
            bottom_ax.clear()
            bottom_ax.set_xlim(-0.05, 1.05)
            bottom_ax.set_ylim(-0.1, 8)
            top_left_ax.clear()
            top_right_ax.clear()
            middle_ax.clear()
            middle_ax.set_xlim(0, N_ITERATIONS)
            middle_ax.set_ylim(0, 1)
            top_left_ax.set_xlim(0, N_ITERATIONS)
            top_left_ax.set_ylim(0, (mc_regret_mean + mc_regret_std).max() + 0.05)
            top_right_ax.set_ylim(
                0,
                (
                    mc_estimated_means_squared_error_mean
                    + mc_estimated_means_squared_error_std
                ).max()
                + 0.5,
            )
            current_params = posterior_params[i].T
            current_means = estimated_means[i]

            for j in range(N_ARMS):
                if current_means[j]:
                    bottom_ax.vlines(
                        x=current_means[j],
                        ymin=0,
                        ymax=0.5,
                        color=COLORS[j],
                        label=f"Est. {j}",
                        linestyle="dashed",
                    )
                x = np.arange(0, 1, 0.01)
                bottom_ax.plot(
                    x,
                    beta.pdf(x, *current_params[j]),
                    color=COLORS[j],
                    label=f"Dist. {j}",
                )
                bottom_ax.scatter(
                    [real_means[j]],
                    [0],
                    color=COLORS[j],
                    label=f"Real {j}",
                )
            x = list(range(i))
            top_left_ax.plot(
                x,
                mc_regret_mean[:i],
                color="red",
                label="Regret",
            )
            top_left_ax.fill_between(
                x,
                mc_regret_mean[:i] + mc_regret_std[:i],
                mc_regret_mean[:i] - mc_regret_std[:i],
                facecolor="red",
                alpha=0.2,
            )
            top_right_ax.plot(
                x,
                mc_estimated_means_squared_error_mean[:i],
                color="blue",
                label="MSE",
            )
            top_right_ax.fill_between(
                x,
                mc_estimated_means_squared_error_mean[:i]
                + mc_estimated_means_squared_error_std[:i],
                mc_estimated_means_squared_error_mean[:i]
                - mc_estimated_means_squared_error_std[:i],
                facecolor="blue",
                alpha=0.2,
            )
            middle_ax.stackplot(
                x,
                *max_rv_distribution_over_iterations[:i].T,
                labels=[f"Arm {i+1}" for i in range(N_ARMS)],
                colors=COLORS,
            )
            middle_ax.set_xlabel("Iterations")
            middle_ax.set_ylabel(r"$P(X_i > X_{-i})$")

            bottom_ax.set_xlabel("Win Rate")
            bottom_ax.set_ylabel("Density")
            top_left_ax.set_ylabel("Regret")
            top_right_ax.set_ylabel("MSE")
            top_right_ax.yaxis.set_label_position("right")

            bottom_ax.legend(loc="upper left", ncol=N_ARMS, prop={"size": 8})
            top_left_ax.legend(loc="upper left", ncol=1, prop={"size": 8})
            top_right_ax.legend(loc="upper right", ncol=1, prop={"size": 8})
            # middle_ax.legend(loc="lower right", ncol=N_ARMS, prop={"size": 8})
            plt.tight_layout()

        ani = FuncAnimation(
            fig,
            animate,
            interval=40,
            blit=False,
            repeat=False,
            frames=N_ITERATIONS - 1,
        )
        plt.title("Optimizing a Bernoulli Multi-Armed Bandit")
        ani.save(
            f"BMAB_optimization_{acqf.__class__.__name__}.gif",
            dpi=300,
            writer=PillowWriter(fps=15),
        )
