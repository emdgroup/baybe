## Example for a Multi Armed Bandit

# This example shows how to use the bernoulli multi armed bandit surrogate.


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from multi_armed_bandit_model import MultiArmedBanditModel
from scipy.stats import bernoulli, beta
from torch import Tensor

from baybe import Campaign
from baybe.acquisition import ThompsonSampling, qUpperConfidenceBound
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

# We are using a 5-armed bandit in this example. The bandit has a random win rate for now.

RENDER = False
N_ARMS = 5
COLORS = ["blue", "red", "green", "black", "pink"]
N_ITERATIONS = 300
acqfs = [
    # Aptimization (best arm identification)
    ThompsonSampling(),
    # Active learning (learning the effect size per arm)
    qUpperConfidenceBound(beta=9999999),
]
acqf = acqfs[0]

np.random.seed(0)

mab = MultiArmedBanditModel(
    real_distributions=[bernoulli(np.random.rand()) for _ in range(N_ARMS)]
)
print("real means", mab.means)


### Campaign

# We are using the BinaryTarget as we are modeling a bernoulli reward.
# The searchspace has one categorical parameter to model the arms of the bandit.

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


### Optimization Loop

real_means = Tensor(mab.means)
total_reward = 0
estimated_means = []
posterior_params = []
regrets = []
mses = []
for i in range(N_ITERATIONS):
    df = campaign.recommend(batch_size=1)

    reward = mab.sample(df.index.tolist())
    total_reward += sum(reward)
    regrets.append((i + 1) * max(real_means) - total_reward)
    df["clicked"] = reward
    campaign.add_measurements(df)

    if i > 0:
        map_means = surrogate.maximum_a_posteriori_per_arm()
        estimated_means.append(map_means)
        posterior_params.append(surrogate._posterior_beta_parameters())
        error = real_means - map_means
        error = error[~error.isnan()]
        mse = (error @ error).mean()
        mses.append(mse.item())

print("real means", real_means)
print("MAP means", surrogate.maximum_a_posteriori_per_arm())
print("optimal expected reward", max(real_means) * N_ITERATIONS)
print("total_reward", total_reward)
print("regret", regrets[-1])
print("MSE", mses[-1])

if RENDER:
    fig, [top_left_ax, bottom_ax] = plt.subplots(2, height_ratios=(0.3, 0.7))
    top_right_ax = top_left_ax.twinx()

    def animate(i):
        """Animate one frame."""
        bottom_ax.clear()
        bottom_ax.set_xlim(-0.05, 1.05)
        bottom_ax.set_ylim(-0.1, 8)
        top_left_ax.clear()
        top_right_ax.clear()
        top_left_ax.set_xlim(0, N_ITERATIONS)
        top_left_ax.set_ylim(0, max(regrets) + 0.5)
        current_params = posterior_params[i]
        current_means = estimated_means[i]

        for j in range(N_ARMS):
            if current_means[j]:
                bottom_ax.vlines(
                    x=current_means[j],
                    ymin=0,
                    ymax=0.5,
                    color=COLORS[j],
                    label=f"MAP {j}",
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
        top_left_ax.plot(
            list(range(len(regrets[:i]))),
            regrets[:i],
            color="red",
            label="Regret",
        )
        top_right_ax.plot(
            list(range(len(mses[:i]))),
            mses[:i],
            color="blue",
            label="MSE",
        )
        bottom_ax.set_xlabel("Win Rate")
        bottom_ax.set_ylabel("Density")
        top_left_ax.set_ylabel("Regret")
        top_right_ax.set_ylabel("MSE")
        top_right_ax.yaxis.set_label_position("right")

        bottom_ax.legend(loc="upper left", ncol=5, prop={"size": 8})
        top_left_ax.legend(loc="upper left", ncol=1, prop={"size": 8})
        top_right_ax.legend(loc="upper right", ncol=1, prop={"size": 8})

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
        "BMAB_optimization.gif",
        dpi=300,
        writer=PillowWriter(fps=15),
    )
