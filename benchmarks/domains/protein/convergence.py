"""Protein DMS single-mutant optimization benchmarks.

One benchmark per dataset, embedding model and batch schedule, optimizing the
measured ``score`` over single-amino-acid mutants (left-skewed datasets are negated
so every benchmark maximizes). Besides score convergence, each iteration also
records instance- and position-retrieval: the proportion of the globally top-X%
scoring mutants (or their positions) selected so far.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from baybe.campaign import Campaign
from baybe.parameters import CustomDiscreteParameter
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_experiment
from baybe.targets import NumericalTarget
from baybe.utils import register_hooks
from benchmarks.definition.base import RunMode, logger
from benchmarks.definition.convergence import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.protein.data import ProteinCaseLoader

DATASETS = [
    "brenan",
    "cas12f",
    "cov2_S",
    "doud",
    "giacomelli",
    "haddox",
    "jones",
    "kelsic",
    "lee",
    "stiffler",
    "zikv_E",
]

EMBEDDING_MODELS = ["ESMpp_small", "ProstT5", "ProtT5XL"]

# Top-fraction thresholds at which retrieval metrics are evaluated.
RETRIEVAL_THRESHOLDS = (0.05, 0.10, 0.20)

REDUCED_FEATURES: int | None = 50

SCHEDULE_SETTINGS: dict[str, ConvergenceBenchmarkSettings] = {
    "batch30_rounds10": ConvergenceBenchmarkSettings(
        batch_size_settings={RunMode.DEFAULT: 30, RunMode.SMOKETEST: 2},
        n_doe_iterations_settings={RunMode.DEFAULT: 10, RunMode.SMOKETEST: 2},
        n_mc_iterations_settings={RunMode.DEFAULT: 30, RunMode.SMOKETEST: 1},
    ),
    "batch60_rounds5": ConvergenceBenchmarkSettings(
        batch_size_settings={RunMode.DEFAULT: 60, RunMode.SMOKETEST: 2},
        n_doe_iterations_settings={RunMode.DEFAULT: 5, RunMode.SMOKETEST: 2},
        n_mc_iterations_settings={RunMode.DEFAULT: 30, RunMode.SMOKETEST: 1},
    ),
}


def _reduce_dimensionality(
    embeddings: pd.DataFrame, n_features: int, seed: int
) -> pd.DataFrame:
    """Reduce the embedding dimensionality via a random Gaussian projection.

    Args:
        embeddings: The embedding matrix indexed by mutation.
        n_features: Target number of features after reduction. Values not smaller
            than the original dimensionality trigger a warning, since no reduction
            then actually takes place.
        seed: Random seed for constructing the projection matrix.

    Returns:
        The reduced embedding matrix indexed by mutation.
    """
    n_original = embeddings.shape[1]
    if n_features >= n_original:
        warnings.warn(
            f"The requested number of features ({n_features}) is not smaller than the "
            f"original number of features ({n_original}); the random projection does "
            f"not reduce the dimensionality.",
            stacklevel=2,
        )
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((n_original, n_features)) / np.sqrt(n_features)
    reduced = embeddings.to_numpy() @ projection
    return pd.DataFrame(
        reduced,
        index=embeddings.index,
        columns=[f"feature_{i}" for i in range(n_features)],
    )


class _RetrievalReference(NamedTuple):
    """Precomputed top-X% reference values for one retrieval threshold."""

    cutoff: float
    """The score above which an instance counts as top-X%."""

    n_top_instances: int
    """The number of instances in the full search space above ``cutoff``."""

    top_positions: frozenset
    """The mutation positions in the full search space above ``cutoff``."""


def compute_retrieval_reference(
    all_scores: npt.ArrayLike,
    all_positions: npt.ArrayLike,
    thresholds: tuple[float, ...] = RETRIEVAL_THRESHOLDS,
) -> dict[float, _RetrievalReference]:
    """Precompute the top-X% cutoffs and positions shared across all iterations.

    Args:
        all_scores: All scores in the full search space.
        all_positions: All mutation positions in the full search space.
        thresholds: Top fractions at which to evaluate retrieval.

    Returns:
        A mapping from threshold to its precomputed reference values.
    """
    all_scores = np.asarray(all_scores)
    all_positions = np.asarray(all_positions)
    reference = {}
    for threshold in thresholds:
        cutoff = np.percentile(all_scores, 100 * (1 - threshold))
        is_top = all_scores >= cutoff
        reference[threshold] = _RetrievalReference(
            cutoff=cutoff,
            n_top_instances=int(is_top.sum()),
            top_positions=frozenset(all_positions[is_top]),
        )
    return reference


def compute_retrieval_metrics(
    selected_scores: npt.ArrayLike,
    selected_positions: npt.ArrayLike,
    reference: dict[float, _RetrievalReference],
) -> dict[str, float]:
    """Compute the top-X% instance and position retrieval for one iteration.

    A position counts as retrieved if at least one selected mutation at that position
    has a score above the top-X% cutoff.

    Args:
        selected_scores: Scores of the instances selected so far.
        selected_positions: Mutation positions of the instances selected so far.
        reference: The top-X% reference values, see :func:`compute_retrieval_reference`.

    Returns:
        A flat mapping from ``"{instance,position}_retrieval_top_{threshold}pct"``
        to its value.
    """
    selected_scores = np.asarray(selected_scores)
    selected_positions = np.asarray(selected_positions)
    metrics = {}
    for threshold, ref in reference.items():
        pct = int(threshold * 100)
        is_top = selected_scores >= ref.cutoff
        metrics[f"instance_retrieval_top_{pct}pct"] = (
            float(is_top.sum() / ref.n_top_instances) if ref.n_top_instances else 0.0
        )
        retrieved_positions = set(selected_positions[is_top]) & ref.top_positions
        metrics[f"position_retrieval_top_{pct}pct"] = (
            float(len(retrieved_positions) / len(ref.top_positions))
            if ref.top_positions
            else 0.0
        )
    return metrics


def _run_dataset(
    dataset: str,
    embedding_model: str,
    settings: ConvergenceBenchmarkSettings,
    n_reduced_features: int | None = None,
) -> pd.DataFrame:
    """Run the protein optimization benchmark for a single dataset.

    Args:
        dataset: Name of the DMS dataset.
        embedding_model: Name of the embedding model to encode the sequences with.
        settings: Configuration settings for the convergence benchmark.
        n_reduced_features: Target number of features for random-projection
            dimensionality reduction of the embeddings. If ``None``, no reduction is
            applied and the full embedding representation is used.

    Returns:
        A dataframe with the score convergence and retrieval metrics per iteration.
    """
    loader = ProteinCaseLoader(case_name=dataset)
    data, embedding_columns = loader.get_aligned_data(embedding_model)

    # Negate left-skewed scores so that every benchmark is a maximization problem.
    sign = -1.0 if loader.metadata.score_skewness < 0 else 1.0
    data["score"] *= sign
    scores = data["score"].to_numpy(float)
    positions = data["mutation_idx"].to_numpy()

    encoding = data.set_index("mutation")[embedding_columns]
    if n_reduced_features is not None:
        encoding = _reduce_dimensionality(
            encoding, n_reduced_features, settings.random_seed
        )
    searchspace = SearchSpace.from_product(
        [CustomDiscreteParameter(name="mutation", data=encoding, decorrelate=False)]
    )
    objective = NumericalTarget(name="score").to_objective()
    templates = {
        "Default Recommender": Campaign(searchspace=searchspace, objective=objective),
        "Random Recommender": Campaign(
            searchspace=searchspace,
            objective=objective,
            recommender=RandomRecommender(),
        ),
    }

    lookup = data[["mutation", "score"]]
    position_by_mutation = dict(zip(data["mutation"], positions))
    reference = compute_retrieval_reference(scores, positions)

    dfs = []
    for recommender, template in templates.items():
        for monte_carlo_run in range(settings.n_mc_iterations):
            seed = settings.random_seed + monte_carlo_run
            retrieval_records: list[dict] = []
            selected_scores: list[float] = []
            selected_positions: list = []

            def track_retrieval(data: pd.DataFrame) -> None:
                """Pre-hook for ``Campaign.add_measurements``."""
                selected_scores.extend(data["score"].tolist())
                selected_positions.extend(
                    data["mutation"].map(position_by_mutation).tolist()
                )
                metrics = compute_retrieval_metrics(
                    selected_scores, selected_positions, reference
                )
                retrieval_records.append(
                    {"Iteration": len(retrieval_records), **metrics}
                )

            original_add_measurements = Campaign.add_measurements
            Campaign.add_measurements = register_hooks(  # type: ignore[method-assign]
                original_add_measurements, pre_hooks=[track_retrieval]
            )
            convergence = simulate_experiment(
                template,
                lookup,
                batch_size=settings.batch_size,
                n_doe_iterations=settings.n_doe_iterations,
                random_seed=seed,
            )
            Campaign.add_measurements = original_add_measurements  # type: ignore[method-assign]
            result = convergence[
                ["Iteration", "Num_Experiments", "score_CumBest"]
            ].merge(pd.DataFrame(retrieval_records), on="Iteration")
            result.insert(0, "Monte_Carlo_Run", monte_carlo_run)
            result.insert(0, "Random_Seed", seed)
            result.insert(0, "Recommender", recommender)
            dfs.append(result)

            logger.info(
                f"[{dataset}/{embedding_model}] completed Monte Carlo run "
                f"{monte_carlo_run + 1}/{settings.n_mc_iterations} for "
                f"'{recommender}' (seed={seed})."
            )
    return pd.concat(dfs, ignore_index=True)


def _make_benchmark_function(
    dataset: str,
    embedding_model: str,
    scenario: str,
    n_reduced_features: int | None = REDUCED_FEATURES,
) -> Callable[[ConvergenceBenchmarkSettings], pd.DataFrame]:
    """Create the benchmark callable for a single dataset, model and schedule.

    Args:
        dataset: Name of the DMS dataset.
        embedding_model: Name of the embedding model to encode the sequences with.
        scenario: Name of the batch schedule, used to disambiguate the benchmark.
        n_reduced_features: Target number of features for random-projection
            dimensionality reduction of the embeddings. If ``None``, no reduction is
            applied and the full embedding representation is used.

    Returns:
        A benchmark function with a dataset-, model- and schedule-specific name and
        docstring.
    """

    def benchmark(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
        return _run_dataset(dataset, embedding_model, settings, n_reduced_features)

    benchmark.__name__ = f"protein_{dataset}_{embedding_model}_{scenario}"
    benchmark.__doc__ = (
        f"Protein DMS single-mutant optimization benchmark on the '{dataset}' "
        f"dataset, using '{embedding_model}' embeddings and the '{scenario}' batch "
        f"schedule. See the module docstring for details on objective, "
        f"recommenders and recorded metrics."
    )
    return benchmark


PROTEIN_BENCHMARKS = [
    ConvergenceBenchmark(
        function=_make_benchmark_function(dataset, embedding_model, scenario),
        settings=settings,
    )
    for dataset in DATASETS
    for embedding_model in EMBEDDING_MODELS
    for scenario, settings in SCHEDULE_SETTINGS.items()
]
