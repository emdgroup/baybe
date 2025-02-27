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

    results = []

    def sample_initial_data():
        p = 0.0001
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

    # Do or do not use stratified outtransform
    for task_stratified_outtransform in [True, False]:
        campaign = Campaign(
            searchspace=searchspace,
            objective=objective,
            recommender=TwoPhaseMetaRecommender(
                recommender=BotorchRecommender(
                    surrogate_model=GaussianProcessSurrogate(
                        task_stratified_outtransform=task_stratified_outtransform,
                    )
                ),
                initial_recommender=RandomRecommender(),
            ),
        )

        results.append(
            simulate_scenarios(
                {f"TL-StratOutTrans{int(task_stratified_outtransform)}": campaign},
                lookup,
                initial_data=[
                    sample_initial_data() for _ in range(settings.n_mc_iterations)
                ],
                batch_size=settings.batch_size,
                n_doe_iterations=settings.n_doe_iterations,
                impute_mode="error",
            )
        )

        # No training data
        results.append(
            simulate_scenarios(
                {
                    "TL-noSource-StratOutTrans"
                    + f"{int(task_stratified_outtransform)}": campaign
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
    n_mc_iterations=10,
)

easom_tl_noise_benchmark = ConvergenceBenchmark(
    function=easom_tl_noise,
    optimal_target_values={"Target": 0.9462931105452647},
    settings=benchmark_config,
)
