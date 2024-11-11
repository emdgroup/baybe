"""Synthetic dataset with three dimensions."""

from uuid import UUID

from numpy import pi, sin, sqrt
from pandas import DataFrame

from baybe.campaign import Campaign
from baybe.objective import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter, NumericalDiscreteParameter
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget, TargetMode
from benchmarks.definition import Benchmark
from benchmarks.definition.config import RecommenderConvergenceAnalysis


def lookup_synthetic_3(z: int, x: float, y: float) -> float:
    """Lookup function for the synthetic dataset."""
    assert -2 * pi <= x <= 2 * pi
    assert -2 * pi <= y <= 2 * pi
    assert z in {1, 2, 3, 4}

    if z == 1:
        return sin(x) * (1 + sin(y))
    if z == 2:
        return x * sin(0.9 * x) + sin(x) * sin(y)
    if z == 3:
        return sqrt(x + 8) * sin(x) + sin(x) * sin(y)
    if z == 4:
        return x * sin(1.666 * sqrt(x + 8)) + sin(x) * sin(y)
    raise ValueError("Invalid z value.")


def synthetic_3(
    scenario_config: RecommenderConvergenceAnalysis,
) -> DataFrame:
    """Synthetic dataset.

    Number of Samples            inf
    Dimensionality                 3
    Features:
        z   discrete   {1,2,3,4}
        x   continuous [-2*pi, 2*pi]
        y   continuous [-2*pi, 2*pi]
    Targets:
        output   continuous
    Best Value 4.09685
    """
    parameters = [
        NumericalContinuousParameter("x", (-2 * pi, 2 * pi)),
        NumericalContinuousParameter("y", (-2 * pi, 2 * pi)),
        NumericalDiscreteParameter("z", (1, 2, 3, 4)),
    ]
    objective = SingleTargetObjective(
        target=NumericalTarget(name="output", mode=TargetMode.MAX)
    )

    scenarios = {}

    for scenario_name, recommender in scenario_config.recommender.items():
        campaign = Campaign(
            searchspace=SearchSpace.from_product(parameters=parameters),
            objective=objective,
            recommender=recommender,
        )
        scenarios[scenario_name] = campaign

    return simulate_scenarios(
        scenarios,
        lookup_synthetic_3,
        batch_size=scenario_config.batch_size,
        n_doe_iterations=scenario_config.n_doe_iterations,
        n_mc_iterations=scenario_config.n_mc_iterations,
        impute_mode="error",
    )


benchmark_config = RecommenderConvergenceAnalysis(
    batch_size=5,
    n_doe_iterations=30,
    n_mc_iterations=50,
    recommender={
        "Random Recommender": RandomRecommender(),
        "Default Recommender": TwoPhaseMetaRecommender(
            RandomRecommender(), BotorchRecommender(), 1
        ),
    },
)

benchmark_synthetic_3 = Benchmark(
    name="Synthetic dataset with three dimensions.",
    settings=benchmark_config,
    identifier=UUID("4e131cb7-4de0-4900-b993-1d7d4a194532"),
    benchmark_callable=synthetic_3,
)
