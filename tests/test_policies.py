"""Test for policies."""

import pandas as pd
import pytest

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import CategoricalParameter, NumericalDiscreteParameter
from baybe.recommenders import BotorchRecommender
from baybe.searchspace import ProductCandidates, SearchSpace
from baybe.searchspace.policies import RandomSamplingPolicy
from baybe.targets import NumericalTarget

p_disc1 = NumericalDiscreteParameter(name="num1", values=(1, 2, 3))
p_disc2 = NumericalDiscreteParameter(name="num2", values=(4, 5, 6))
p_cat = CategoricalParameter("cat", ("a", "b", "c"))


@pytest.mark.parametrize(
    ("parameters"),
    [
        pytest.param((p_disc1, p_disc2), id="num_disc"),
        pytest.param((p_disc1, p_cat), id="num_disc_cat"),
    ],
)
def test_random_sampling_policy(parameters):
    """The RandomSamplingPolicy returns a subset of candidates of the correct size."""
    # Create a simple candidate space
    candidates = ProductCandidates(parameters=parameters)

    # Create the policy
    n_sample = 2
    policy = RandomSamplingPolicy(n=n_sample, seed=42)

    sampled_candidates = policy(candidates)

    assert sampled_candidates.to_lazy().collect().shape[0] == n_sample

    # Seed is correctly applied
    sampled_candidates_1 = policy(candidates)
    assert (
        sampled_candidates.to_lazy()
        .collect()
        .to_pandas()
        .equals(sampled_candidates_1.to_lazy().collect().to_pandas())
    )


def test_call_chain():
    """The policy is applied before recommend."""
    # TODO: Test more candidate types and policies when available

    p1 = NumericalDiscreteParameter(name="p1", values=(1, 2, 3, 4, 5))
    p2 = NumericalDiscreteParameter(name="p2", values=(1, 2, 3, 4, 5))
    measurements = pd.DataFrame(
        {
            "p1": [1, 2, 4],
            "p2": [3, 5, 1],
            "target": [0.1, 0.5, 0.9],
        }
    )
    searchspace = SearchSpace.from_product(parameters=(p1, p2))

    objective = SingleTargetObjective(
        target=NumericalTarget(name="target", minimize=False)
    )
    recommender = BotorchRecommender()

    campaign = Campaign(
        searchspace=searchspace, objective=objective, recommender=recommender
    )

    campaign.add_measurements(measurements)
    n_sample = 10
    policy = RandomSamplingPolicy(n=n_sample, seed=42)
    candidates = searchspace.discrete.get_candidates(policy)
    assert candidates.shape[0] == n_sample

    recommendations = campaign.recommend(batch_size=2, policy=policy)

    # The recommendations are a subset of the candidates
    assert all(rec in candidates.to_dict() for rec in recommendations.to_dict())


def test_policy_chain():
    """The policy is applied before recommend."""
    ...


def test_recommender_policy():
    """The recommender is used as a policy."""
    ...
