"""Synthetic dataset. Custom parabolic test with irrelevant parameters."""

from uuid import UUID

from numpy import sin, sqrt
from pandas import DataFrame

from baybe.campaign import Campaign
from baybe.objective import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter, NumericalDiscreteParameter
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget, TargetMode
from benchmark.src import SingleExecutionBenchmark


def lookup_synthetic_3(*args) -> float:
    """Synthetic dataset. Hand-crafted by Alexander Hopp.

    ================================================================================
                                        Summary
    --------------------------------------------------------------------------------
        Number of Samples            inf
        Dimensionality                 3
        Features:
            x   continuous [-2*pi, 2*pi]
            y   continuous [-2*pi, 2*pi]
            z   discrete {1,2,3,4}
        Targets:
            output   continuous
    ================================================================================
    Best Value 4.09685
    """
    (x, y, z) = args
    return (
        (z == 1) * sin(x) * (1 + sin(y))
        + (z == 2) * (x * sin(0.9 * x) + sin(x) * sin(y))
        + (z == 3) * (sqrt(x + 8) * sin(x) + sin(x) * sin(y))
        + (z == 4) * (x * sin(1.666 * sqrt(x + 8)) + sin(x) * sin(y))
    )


def synthetic_3() -> tuple[DataFrame, dict[str, str]]:
    """Synthetic dataset. Custom parabolic test with irrelevant parameters."""
    synthetic_3_continues = [
        NumericalContinuousParameter("x", (-6.283185, 6.283185)),
        NumericalContinuousParameter("y", (-6.283185, 6.283185)),
        NumericalDiscreteParameter("z", (1, 2, 3, 4)),
    ]

    objective = SingleTargetObjective(
        target=NumericalTarget(name="output", mode=TargetMode.MAX)
    )

    campaign = Campaign(
        searchspace=SearchSpace.from_product(parameters=synthetic_3_continues),
        objective=objective,
    )
    campaign_rand = Campaign(
        searchspace=SearchSpace.from_product(parameters=synthetic_3_continues),
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
        lookup_synthetic_3,
        batch_size=batch_size,
        n_doe_iterations=n_doe_iterations,
        n_mc_iterations=n_mc_iterations,
        impute_mode="error",
    ), metadata


benchmark_synthetic_3 = SingleExecutionBenchmark(
    title="Synthetic dataset. Hand-crafted by Alexander Hopp.",
    identifier=UUID("4e131cb7-4de0-4900-b993-1d7d4a194532"),
    benchmark_function=synthetic_3,
)
