"""Protein DMS single-mutant optimization benchmarks.

One convergence benchmark is defined per DMS dataset in ``benchmarks/data/protein``.
Each benchmark optimizes the measured ``score`` over the pool of single-amino-acid
mutants, using mean-pooled ``ESMpp_small`` embeddings as the computational
representation of the sequences.

Two protein-specific evaluation metrics are recorded per iteration in addition to the
score convergence:

- Instance retrieval: proportion of the globally top-X% scoring mutants selected so far.
- Position retrieval: proportion of the globally top-X% scoring mutation positions
  selected so far (by evaluating retrieval of top scoring mutants across positions).

Datasets whose score distribution is left-skewed (negative skewness stored in the
per-dataset ``metadata.json``) are negated so that every benchmark is a maximization
problem on right-skewed distribution.

Each dataset is optimized with both the default BayBE recommender and a random
baseline, under two equal-budget batch schedules (many small vs. few large batches).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from copy import deepcopy

import numpy as np
import pandas as pd

from baybe.campaign import Campaign
from baybe.parameters import CustomDiscreteParameter
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.settings import Settings
from baybe.targets import NumericalTarget
from benchmarks.data.utils import DATA_PATH
from benchmarks.definition.base import RunMode
from benchmarks.definition.convergence import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)

# The DMS datasets available under ``benchmarks/data/protein``.
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

# Embedding model used to encode the sequences.
EMBEDDING_MODEL = "ESMpp_small"

# Top-fraction thresholds at which retrieval metrics are evaluated.
RETRIEVAL_THRESHOLDS = (0.05, 0.10, 0.20)

# Target number of features for random-projection dimensionality reduction of the
# embeddings, applied to all datasets. Set to ``None`` to disable the reduction and
# use the full embedding representation.
REDUCED_FEATURES: int | None = 50


def _load_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Load the mutation table, embeddings and score skewness for a dataset.

    Args:
        dataset: Name of the DMS dataset.

    Returns:
        The mutation table, the aligned embedding matrix and the score skewness.
    """
    directory = DATA_PATH / "protein" / dataset
    mutations = pd.read_csv(directory / "mutations.tsv", sep="\t")
    embeddings = pd.read_parquet(directory / f"embeddings_{EMBEDDING_MODEL}.parquet")
    with open(directory / "metadata.json") as file:
        skewness = json.load(file)["score_skewness"]
    return mutations, embeddings, skewness


def _reduce_dimensionality(
    embeddings: pd.DataFrame, n_features: int, seed: int
) -> pd.DataFrame:
    """Reduce the embedding dimensionality via a random Gaussian projection.

    Multiplies the ``(n_sequences, n_original_features)`` embedding matrix by a random
    ``(n_original_features, n_features)`` Gaussian matrix, yielding a lower-dimensional
    representation in the style of randomized linear algebra. The projection is always
    applied; if the requested number of features is not smaller than the current one,
    a warning is emitted since the representation is not actually reduced.

    Args:
        embeddings: The embedding matrix indexed by sequence.
        n_features: Target number of features after reduction.
        seed: Random seed for constructing the projection matrix.

    Returns:
        The reduced embedding matrix indexed by sequence.
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


def compute_instance_retrieval(
    selected_scores: np.ndarray,
    all_scores: np.ndarray,
    thresholds: tuple[float, ...] = RETRIEVAL_THRESHOLDS,
) -> dict[float, float]:
    """Compute the proportion of top-X% instances retrieved.

    Args:
        selected_scores: Scores of the instances selected so far.
        all_scores: All scores in the full search space.
        thresholds: Top fractions at which to evaluate retrieval.

    Returns:
        A mapping from threshold to the proportion of top instances retrieved.
    """
    all_scores = np.asarray(all_scores)
    selected_scores = np.asarray(selected_scores)
    result = {}
    for threshold in thresholds:
        cutoff = np.percentile(all_scores, 100 * (1 - threshold))
        n_top = int((all_scores >= cutoff).sum())
        if n_top == 0:
            result[threshold] = 0.0
        else:
            result[threshold] = float((selected_scores >= cutoff).sum() / n_top)
    return result


def compute_position_retrieval(
    selected_scores: np.ndarray,
    selected_positions: np.ndarray,
    all_scores: np.ndarray,
    all_positions: np.ndarray,
    thresholds: tuple[float, ...] = RETRIEVAL_THRESHOLDS,
) -> dict[float, float]:
    """Compute the proportion of top-X% mutation positions retrieved.

    A position counts as retrieved if at least one selected mutation at that position
    has a score above the top-X% cutoff.

    Args:
        selected_scores: Scores of the instances selected so far.
        selected_positions: Mutation positions of the instances selected so far.
        all_scores: All scores in the full search space.
        all_positions: All mutation positions in the full search space.
        thresholds: Top fractions at which to evaluate retrieval.

    Returns:
        A mapping from threshold to the proportion of top positions retrieved.
    """
    all_scores = np.asarray(all_scores)
    all_positions = np.asarray(all_positions)
    selected_scores = np.asarray(selected_scores)
    selected_positions = np.asarray(selected_positions)
    result = {}
    for threshold in thresholds:
        cutoff = np.percentile(all_scores, 100 * (1 - threshold))
        top_positions = set(all_positions[all_scores >= cutoff])
        if len(top_positions) == 0:
            result[threshold] = 0.0
        else:
            retrieved = set(selected_positions[selected_scores >= cutoff])
            retrieved &= top_positions
            result[threshold] = float(len(retrieved) / len(top_positions))
    return result


def _budget_schedules(
    settings: ConvergenceBenchmarkSettings,
) -> dict[str, tuple[int, int]]:
    """Get the equal-budget batch schedules for the current run mode.

    The primary schedule comes from the settings. For non-smoke runs a second,
    equal-budget schedule is added that doubles the batch size and halves the number
    of rounds (trading more rounds for larger batches).

    Args:
        settings: Configuration settings for the convergence benchmark.

    Returns:
        A mapping from scenario label to ``(batch_size, n_rounds)``.

    Raises:
        ValueError: If the equal-budget second schedule cannot be constructed because
            the number of rounds is odd.
    """
    batch_size = settings.batch_size
    n_rounds = settings.n_doe_iterations
    schedules = {f"batch{batch_size}_rounds{n_rounds}": (batch_size, n_rounds)}
    if settings.runmode is not RunMode.SMOKETEST:
        budget = batch_size * n_rounds
        doubled_batch = batch_size * 2
        halved_rounds = n_rounds // 2
        if doubled_batch * halved_rounds != budget:
            raise ValueError(
                f"Cannot construct an equal-budget schedule with double the batch "
                f"size: the total budget {budget=} is not divisible by the doubled "
                f"batch size {doubled_batch=}. This requires an even number of "
                f"rounds (got {n_rounds=})."
            )
        schedules[f"batch{doubled_batch}_rounds{halved_rounds}"] = (
            doubled_batch,
            halved_rounds,
        )
    return schedules


def _run_dataset(
    dataset: str,
    settings: ConvergenceBenchmarkSettings,
    n_reduced_features: int | None = None,
) -> pd.DataFrame:
    """Run the protein optimization benchmark for a single dataset.

    Args:
        dataset: Name of the DMS dataset.
        settings: Configuration settings for the convergence benchmark.
        n_reduced_features: Target number of features for random-projection
            dimensionality reduction of the embeddings. If ``None``, no reduction is
            applied and the full embedding representation is used.

    Returns:
        A dataframe with the score convergence and retrieval metrics per iteration.
    """
    mutations, embeddings, skewness = _load_dataset(dataset)

    # Negate left-skewed scores so that every benchmark is a maximization problem.
    sign = -1.0 if skewness < 0 else 1.0
    scores = mutations["score"].to_numpy(float) * sign
    positions = mutations["mutation_idx"].to_numpy()
    sequences = mutations["seq"].to_numpy()

    encoding = embeddings.set_axis(sequences, axis="index")
    if n_reduced_features is not None:
        encoding = _reduce_dimensionality(
            encoding, n_reduced_features, settings.random_seed
        )
    searchspace = SearchSpace.from_product(
        [CustomDiscreteParameter(name="seq", data=encoding, decorrelate=False)]
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

    lookup = pd.DataFrame({"seq": sequences, "score": scores})
    position_by_sequence = dict(zip(sequences, positions))
    schedules = _budget_schedules(settings)

    records = []
    for recommender, template in templates.items():
        for scenario, (batch_size, n_rounds) in schedules.items():
            for monte_carlo_run in range(settings.n_mc_iterations):
                seed = settings.random_seed + monte_carlo_run
                with Settings(random_seed=seed):
                    campaign = deepcopy(template)
                    selected_scores: list[float] = []
                    selected_positions: list = []
                    n_experiments = 0
                    for iteration in range(n_rounds):
                        recommendation = campaign.recommend(batch_size=batch_size)
                        measured = recommendation.merge(lookup, on="seq", how="left")
                        campaign.add_measurements(measured[["seq", "score"]])

                        selected_scores.extend(measured["score"].tolist())
                        selected_positions.extend(
                            position_by_sequence[sequence]
                            for sequence in measured["seq"]
                        )
                        n_experiments += len(measured)

                        instance = compute_instance_retrieval(selected_scores, scores)
                        position = compute_position_retrieval(
                            selected_scores, selected_positions, scores, positions
                        )
                        records.append(
                            {
                                "Recommender": recommender,
                                "Scenario": scenario,
                                "Random_Seed": seed,
                                "Monte_Carlo_Run": monte_carlo_run,
                                "Iteration": iteration,
                                "Num_Experiments": n_experiments,
                                "score_CumBest": max(selected_scores),
                                **{
                                    f"instance_retrieval_top_{int(t * 100)}pct": (
                                        instance[t]
                                    )
                                    for t in RETRIEVAL_THRESHOLDS
                                },
                                **{
                                    f"position_retrieval_top_{int(t * 100)}pct": (
                                        position[t]
                                    )
                                    for t in RETRIEVAL_THRESHOLDS
                                },
                            }
                        )
    return pd.DataFrame(records)


def _make_benchmark_function(
    dataset: str,
    n_reduced_features: int | None = REDUCED_FEATURES,
) -> Callable[[ConvergenceBenchmarkSettings], pd.DataFrame]:
    """Create the benchmark callable for a single dataset.

    Args:
        dataset: Name of the DMS dataset.
        n_reduced_features: Target number of features for random-projection
            dimensionality reduction of the embeddings. If ``None``, no reduction is
            applied and the full embedding representation is used.

    Returns:
        A benchmark function with a dataset-specific name and docstring.
    """

    def benchmark(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
        return _run_dataset(dataset, settings, n_reduced_features)

    benchmark.__name__ = f"protein_{dataset}"
    benchmark.__doc__ = (
        f"Protein DMS single-mutant optimization benchmark on the '{dataset}' "
        f"dataset.\n\n"
        f"    Key characteristics:\n"
        f"    • Search space: single-amino-acid mutants encoded via\n"
        f"      {EMBEDDING_MODEL} mean-pooled embeddings.\n"
        f"    • Objective: maximization of the measured score (negated for\n"
        f"      left-skewed datasets).\n"
        f"    • Recommenders: default BayBE and a random baseline.\n"
        f"    • Recorded metrics: score convergence, instance retrieval and\n"
        f"      position retrieval at the top {RETRIEVAL_THRESHOLDS} fractions.\n"
    )
    return benchmark


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size_settings={
        RunMode.DEFAULT: 30,
        RunMode.SMOKETEST: 2,
    },
    n_doe_iterations_settings={
        RunMode.DEFAULT: 10,
        RunMode.SMOKETEST: 2,
    },
    n_mc_iterations_settings={
        RunMode.DEFAULT: 30,
        RunMode.SMOKETEST: 1,
    },
)

PROTEIN_BENCHMARKS = [
    ConvergenceBenchmark(
        function=_make_benchmark_function(dataset, n_reduced_features=REDUCED_FEATURES),
        settings=benchmark_config,
    )
    for dataset in DATASETS
]
