"""Synthetic dataset. Custom parabolic test with irrelevant parameters."""

from uuid import UUID

from pandas import DataFrame

from baybe.campaign import Campaign
from baybe.objective import SingleTargetObjective
from baybe.parameters import (
    NumericalContinuousParameter,
)
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget, TargetMode
from benchmark.src import SingleExecutionBenchmark


def lookup_synthetic_1(*args) -> float:
    """Synthetic dataset. Custom parabolic test with irrelevant parameters.

    ================================================================================
                                        Summary
    --------------------------------------------------------------------------------
        Number of Samples            inf
        Dimensionality                20
        Features:
            x1   continuous [0, 1]
            x3   continuous [0, 1]
            x6   continuous [0, 1]
            x12  continuous [0, 1]
            x18  continuous [0, 1]
            15 other xi's are irrelevant
        Targets:
            output   continuous
    ================================================================================
    Best Value 0.19999999999999996
    """
    (x1, _, x3, _, _, x6, _, _, _, _, _, x12, _, _, _, _, _, x18, _, _) = args
    return (
        (x1 - 2) ** 2
        + 0.8 * x3
        + 0.4 * (x6 + 1) ** 2
        + (x12) ** 2
        - 0.3 * (x18 - 2) ** 2
        + x1 * x6
    )


def synthetic_1() -> tuple[DataFrame, dict[str, str]]:
    """Synthetic dataset. Custom parabolic test with irrelevant parameters."""
    synthetic_1_continues = [
        NumericalContinuousParameter("x1", (0, 1)),
        NumericalContinuousParameter("x2", (0, 1)),
        NumericalContinuousParameter("x3", (0, 1)),
        NumericalContinuousParameter("x4", (0, 1)),
        NumericalContinuousParameter("x5", (0, 1)),
        NumericalContinuousParameter("x6", (0, 1)),
        NumericalContinuousParameter("x7", (0, 1)),
        NumericalContinuousParameter("x8", (0, 1)),
        NumericalContinuousParameter("x9", (0, 1)),
        NumericalContinuousParameter("x10", (0, 1)),
        NumericalContinuousParameter("x11", (0, 1)),
        NumericalContinuousParameter("x12", (0, 1)),
        NumericalContinuousParameter("x13", (0, 1)),
        NumericalContinuousParameter("x14", (0, 1)),
        NumericalContinuousParameter("x15", (0, 1)),
        NumericalContinuousParameter("x16", (0, 1)),
        NumericalContinuousParameter("x17", (0, 1)),
        NumericalContinuousParameter("x18", (0, 1)),
        NumericalContinuousParameter("x19", (0, 1)),
        NumericalContinuousParameter("x20", (0, 1)),
    ]

    objective = SingleTargetObjective(
        target=NumericalTarget(name="output", mode=TargetMode.MIN)
    )

    campaign = Campaign(
        searchspace=SearchSpace.from_product(parameters=synthetic_1_continues),
        objective=objective,
    )
    campaign_rand = Campaign(
        searchspace=SearchSpace.from_product(parameters=synthetic_1_continues),
        recommender=RandomRecommender(),
        objective=objective,
    )

    batch_size = 5
    n_doe_iterations = 30
    n_mc_iterations = 50

    metadata = {
        "DOE_iterations": str(n_doe_iterations),
        "batch_size": str(batch_size),
        "n_mc_iterations": str(n_mc_iterations),
    }

    scenarios = {
        "Default Two Phase Meta Recommender": campaign,
        "Random Baseline": campaign_rand,
    }
    return simulate_scenarios(
        scenarios,
        lookup_synthetic_1,
        batch_size=batch_size,
        n_doe_iterations=n_doe_iterations,
        n_mc_iterations=n_mc_iterations,
        impute_mode="error",
    ), metadata


benchmark_synthetic_1 = SingleExecutionBenchmark(
    title="Custom parabolic test with irrelevant parameters.",
    identifier=UUID("7a855462-048a-4a1e-86b3-ac8fc6bd91aa"),
    benchmark_function=synthetic_1,
)
