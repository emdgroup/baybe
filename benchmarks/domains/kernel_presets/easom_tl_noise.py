"""Transfer learning benchmark with noisy Easom functions as tasks."""

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
from benchmarks.domains.easom_tl_noise import space_data


def easom_tl_noise(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Benchmark function comparing TL and non-TL campaigns.

    Inputs:
        x0  Discrete numerical parameter [-100,100]
        x1  Discrete numerical parameter [-100,100]
        Function  Discrete task parameter
    Output: continuous
    Objective: Maximization
    Optimal Inputs: [
        {
            x0 3.006012
            x1 3.006012
        }
    ]
    Optimal Output: 0.9462931105452647
    """
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

    def sample_initial_data():
        p = 0.0005
        upsample_max_thr = 0.5
        n_upsample_max = 3
        return pd.concat(
            [
                # Sample specific fraction of initial data
                initial_data.sample(frac=p),
                # Add some points near optimum
                initial_data.query(
                    f"{objective._target.name}>{upsample_max_thr}"
                ).sample(n=n_upsample_max),
            ]
        )

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
                sample_initial_data() for _ in range(settings.n_mc_iterations)
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
                sample_initial_data() for _ in range(settings.n_mc_iterations)
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
    n_mc_iterations=100,
)

easom_tl_noise_benchmark = ConvergenceBenchmark(
    function=easom_tl_noise,
    optimal_target_values={"Target": 0.9462931105452647},
    settings=benchmark_config,
)
