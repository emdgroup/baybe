"""Botorch kernel preset benchmark with inverted Hartmann functions as tasks."""

from __future__ import annotations

import pandas as pd

from baybe.campaign import Campaign
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.simulation import simulate_scenarios
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.presets import BotorchKernelFactory
from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.hartmann_tl_inverted_noise import space_data


def hartmann_tl_inverted_noise(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Benchmark function comparing TL and non-TL campaigns.

    Inputs:
        x0  Discrete numerical parameter [0,1]
        x1  Discrete numerical parameter [0,1]
        x2  Discrete numerical parameter [0,1]
        Function  Discrete task parameter
    Output: continuous
    Objective: Maximization
    Optimal Inputs: [
        {
            x0 0.25
            x1 0.6
            x2 0.75
        }
    ]
    Optimal Output: 2.999716768817375
    """
    print("hartmann_tl_inverted_noise")

    objective, searchspace, searchspace_nontl, initial_data, lookup = space_data()

    recommender_botorch_preset = TwoPhaseMetaRecommender(
        recommender=BotorchRecommender(
            surrogate_model=GaussianProcessSurrogate(
                kernel_or_factory=BotorchKernelFactory()
            )
        ),
        initial_recommender=RandomRecommender(),
    )

    results = []

    # TL
    results.append(
        simulate_scenarios(
            {
                "TL": Campaign(
                    searchspace=searchspace,
                    objective=objective,
                )
            },
            lookup,
            initial_data=[
                initial_data.sample(frac=0.1) for _ in range(settings.n_mc_iterations)
            ],
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            impute_mode="error",
        )
    )
    # TL with botorch preset
    results.append(
        simulate_scenarios(
            {
                "TL-botorchPreset": Campaign(
                    searchspace=searchspace,
                    objective=objective,
                    recommender=recommender_botorch_preset,
                )
            },
            lookup,
            initial_data=[
                initial_data.sample(frac=0.1) for _ in range(settings.n_mc_iterations)
            ],
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            impute_mode="error",
        )
    )

    # Non-TL campaign
    results.append(
        simulate_scenarios(
            {"nonTL": Campaign(searchspace=searchspace_nontl, objective=objective)},
            lookup,
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            n_mc_iterations=settings.n_mc_iterations,
            impute_mode="error",
        )
    )
    # Non-TL campaign with botorch preset
    results.append(
        simulate_scenarios(
            {
                "nonTL-botorchPreset": Campaign(
                    searchspace=searchspace_nontl,
                    objective=objective,
                    recommender=recommender_botorch_preset,
                )
            },
            lookup,
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            n_mc_iterations=settings.n_mc_iterations,
            impute_mode="error",
        )
    )

    results = pd.concat(results)
    return results


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=10,
    n_mc_iterations=50,
)

hartmann_tl_inverted_noise_benchmark = ConvergenceBenchmark(
    function=hartmann_tl_inverted_noise,
    optimal_target_values={"Target": 2.999716768817375},
    settings=benchmark_config,
)
