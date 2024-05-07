import click
import numpy as np
from multi_armed_bandit import MultiArmedBandit
from scipy.stats import bernoulli

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

np.random.seed(0)


@click.command()
@click.option("--n_arms", default=2, help="Number of arms")
@click.option("--n_iter", default=1, help="Number of iterations")
@click.option(
    "--log_each_n_steps", default=100, help="Number of iteratins between prints."
)
def run_experiment(n_arms, n_iter, log_each_n_steps):
    """Search for the arm with maximum win rate."""
    mab = MultiArmedBandit(
        real_distributions=[bernoulli(np.random.rand()) for _ in range(n_arms)]
    )
    print("real means", mab.means)
    target = BinaryTarget(name="win_rate")
    objective = SingleTargetObjective(target=target)
    parameters = [
        CategoricalParameter(
            name="arm",
            values=[str(i) for i in range(n_arms)],
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

    total_reward = 0
    for i in range(n_iter):
        df = campaign.recommend(batch_size=1)
        reward = mab.sample(df.index.tolist())
        total_reward += sum(reward)
        df["win_rate"] = reward
        campaign.add_measurements(df)

        if (i + 1) % log_each_n_steps == 0:
            print("iter", i + 1)
            print("estimated means", mabs.means)
            print("-" * 5)

    real_means = mab.means
    print("real means", real_means)
    print("optimal expected reward", max(real_means) * n_iter)
    print("total_reward", total_reward)
    print("mean reward", total_reward / n_iter)


if __name__ == "__main__":
    run_experiment()
