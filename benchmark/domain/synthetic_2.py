"""Synthetic dataset. Michaelewicz function in 5d with parameter m=10."""

from uuid import UUID

from numpy import pi, sin
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


def lookup_synthetic_2(*args) -> float:
    """Synthetic dataset. Michaelewicz function in 5d with parameter m=10.

    ================================================================================
                                        Summary
    --------------------------------------------------------------------------------
        Number of Samples            inf
        Dimensionality                 5
        Features:
            x1   continuous [0, pi]
            x2   continuous [0, pi]
            x3   continuous [0, pi]
            x4   continuous [0, pi]
            x5   continuous [0, pi]
        Targets:
            output   continuous
    ================================================================================
    Best Value -4.687658
    """
    (x1, x2, x3, x4, x5) = args
    return -(
        sin(x1) * sin(1 * x1**2 / pi) ** (2 * 10)
        + sin(x2) * sin(2 * x2**2 / pi) ** (2 * 10)
        + sin(x3) * sin(3 * x3**2 / pi) ** (2 * 10)
        + sin(x4) * sin(4 * x4**2 / pi) ** (2 * 10)
        + sin(x5) * sin(5 * x5**2 / pi) ** (2 * 10)
    )


def synthetic_2() -> tuple[DataFrame, dict[str, str]]:
    """Synthetic dataset. Michaelewicz function in 5d with parameter m=10."""
    synthetic_2_continues = [
        NumericalContinuousParameter("x1", (0, 3.14159)),
        NumericalContinuousParameter("x2", (0, 3.14159)),
        NumericalContinuousParameter("x3", (0, 3.14159)),
        NumericalContinuousParameter("x4", (0, 3.14159)),
        NumericalContinuousParameter("x5", (0, 3.14159)),
    ]

    objective = SingleTargetObjective(
        target=NumericalTarget(name="output", mode=TargetMode.MIN)
    )

    campaign = Campaign(
        searchspace=SearchSpace.from_product(parameters=synthetic_2_continues),
        objective=objective,
    )
    campaign_rand = Campaign(
        searchspace=SearchSpace.from_product(parameters=synthetic_2_continues),
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
        lookup_synthetic_2,
        batch_size=batch_size,
        n_doe_iterations=n_doe_iterations,
        n_mc_iterations=n_mc_iterations,
        impute_mode="error",
    ), metadata


benchmark_synthetic_2 = SingleExecutionBenchmark(
    title="Michaelewicz function in 5d with parameter m=10.",
    identifier=UUID("9b730834-d531-4d65-8842-7fdc1c0a2506"),
    benchmark_function=synthetic_2,
)
