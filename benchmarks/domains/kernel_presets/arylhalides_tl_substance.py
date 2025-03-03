"""Benchmark on ArylHalides data with two distinct arylhalides as TL tasks."""

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
from benchmarks.domains.arylhalides_tl_substance import space_data


def arylhalides_tl_substance(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Benchmark function comparing TL and non-TL campaigns.

    Inputs:
        base  Discrete substance with numerical encoding
        ligand  Discrete substance with numerical encoding
        additive    Discrete substance with numerical encoding
        Concentration   Continuous
        aryl_halide  Discrete task parameter
    Output: continuous
    Objective: Maximization
    Optimal Inputs: [
        {
            base    MTBD
            ligand  AdBrettPhos
            additive N,N-dibenzylisoxazol-3-amine
        }
    ]
    Optimal Output: 68.24812709999999
    """
    print("arylhalides_tl_substance")

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

arylhalides_tl_substance_benchmark = ConvergenceBenchmark(
    function=arylhalides_tl_substance,
    optimal_target_values={"yield": 68.24812709999999},
    settings=benchmark_config,
)
