"""Tests for the TFPR recommender."""

from __future__ import annotations

import math
from typing import Any, NoReturn

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from baybe import Campaign
from baybe.exceptions import IncompatibilityError
from baybe.objectives import ParetoObjective
from baybe.objectives.base import Objective
from baybe.parameters import NumericalContinuousParameter, NumericalDiscreteParameter
from baybe.recommenders import (
    RandomRecommender,
    TFPRRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.recommenders.pure.tfpr import _EPSILON, _auto_top_fraction, _tfpr_fitness
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget


class FakePosteriorStatsSurrogate:
    """Surrogate returning deterministic posterior statistics from parameter values."""

    def __init__(self) -> None:
        self.candidates: pd.DataFrame | None = None
        self.measurements: pd.DataFrame | None = None

    def fit(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        """Record the training data."""
        self.measurements = measurements.copy()

    def posterior_stats(
        self,
        candidates: pd.DataFrame,
        stats: tuple[str, str] = ("mean", "std"),
    ) -> pd.DataFrame:
        """Return deterministic posterior means and standard deviations."""
        assert stats == ("mean", "std")
        self.candidates = candidates.copy()
        p = candidates["p"].astype(float)
        return pd.DataFrame(
            {
                "yield_mean": p,
                "yield_std": np.where(p == 1.0, 10.0, 0.0),
                "cost_mean": -p,
                "cost_std": 0.0,
            },
            index=candidates.index,
        )

    def to_botorch(self) -> NoReturn:
        """Block unused conversion to BoTorch."""
        raise NotImplementedError


class FakeUnsupportedSurrogate:
    """Surrogate that records unexpected fitting and lacks posterior statistics."""

    def __init__(self) -> None:
        self.was_fit = False

    def fit(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        """Record an unexpected fit call."""
        self.was_fit = True

    def to_botorch(self) -> NoReturn:
        """Block unused conversion to BoTorch."""
        raise NotImplementedError


def _dense_tfpr_fitness(
    values: np.ndarray,
    weights: np.ndarray,
    tolerances: np.ndarray,
    top_fraction: float | None,
) -> np.ndarray:
    """Compute dense-matrix TFPR fitness for comparison in tests."""
    n_candidates, n_objectives = values.shape
    counts = np.zeros((n_candidates, n_candidates), dtype=float)
    fraction = (
        _auto_top_fraction(n_candidates) if top_fraction is None else top_fraction
    )
    top_k = min(n_candidates, max(2, math.ceil(n_candidates * fraction)))

    for objective_index in range(n_objectives):
        weight = weights[objective_index]
        if weight == 0:
            continue
        order = np.argsort(-values[:, objective_index], kind="stable")[:top_k]
        for rank, left in enumerate(order):
            for right in order[rank + 1 :]:
                left_value = values[left, objective_index]
                right_value = values[right, objective_index]
                difference = abs(left_value - right_value)
                scale = max(abs(left_value), abs(right_value))
                tolerance = tolerances[objective_index]
                if left_value == right_value or difference <= tolerance * scale:
                    counts[left, right] += weight / 2
                    counts[right, left] += weight / 2
                else:
                    counts[left, right] += weight

    total_weight = weights.sum()
    threshold = 0.5 * total_weight
    fitness = np.zeros(n_candidates, dtype=float)
    for candidate_index in range(n_candidates):
        others = np.arange(n_candidates) != candidate_index
        mean_dominance = counts[candidate_index].sum() / (
            (n_candidates - 1) * total_weight
        )
        n_dominating = np.count_nonzero(counts[candidate_index, others] > threshold)
        n_submitting = np.count_nonzero(counts[candidate_index, others] < threshold)
        fitness[candidate_index] = (
            mean_dominance * (n_dominating + _EPSILON) / (n_submitting + _EPSILON)
        )
    return fitness


@pytest.fixture(name="pareto_objective")
def fixture_pareto_objective() -> ParetoObjective:
    """A two-target Pareto objective with one minimization target."""
    return ParetoObjective(
        [NumericalTarget("yield"), NumericalTarget("cost", minimize=True)]
    )


@pytest.fixture(name="searchspace")
def fixture_searchspace() -> SearchSpace:
    """A small discrete search space."""
    return NumericalDiscreteParameter("p", [0, 1, 2, 3]).to_searchspace()


@pytest.fixture(name="measurements")
def fixture_measurements() -> pd.DataFrame:
    """Nonempty measurements for the TFPR surrogate."""
    return pd.DataFrame({"p": [3], "yield": [3.0], "cost": [-3.0]})


@pytest.mark.parametrize(
    ("n_candidates", "expected"),
    [
        pytest.param(5000, 1.0, id="lower_boundary"),
        pytest.param(5001, 0.923_713_527_2, id="lower_middle"),
        pytest.param(20000, 0.743_342_959_3, id="upper_middle"),
        pytest.param(20001, 0.2, id="upper_boundary"),
    ],
)
def test_auto_top_fraction(n_candidates: int, expected: float) -> None:
    """The automatic top fraction follows the original size-dependent rule."""
    assert _auto_top_fraction(n_candidates) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("values", "top_fraction", "expected"),
    [
        pytest.param(
            [[3.0], [2.0], [1.0]],
            1.0,
            [41.0, 0.5, 0.0],
            id="known_ranking",
        ),
        pytest.param(
            [[1.0], [1.0]],
            1.0,
            [0.5, 0.5],
            id="exact_tie",
        ),
        pytest.param(
            [[4.0], [3.0], [2.0], [1.0]],
            0.5,
            [1.05 / (3 * 2.05), 0.0, 0.0, 0.0],
            id="top_fraction",
        ),
    ],
)
def test_tfpr_fitness_matches_hand_calculation(
    values: list[list[float]], top_fraction: float, expected: list[float]
) -> None:
    """TFPR fitness matches values calculated directly from its definition."""
    actual = _tfpr_fitness(
        np.array(values), np.array([1]), np.array([0.0]), top_fraction
    )

    assert_allclose(actual, expected)


def test_tfpr_fitness_uses_automatic_top_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting the top fraction activates the automatic rule."""

    def auto_top_fraction(n_candidates: int) -> float:
        assert n_candidates == 4
        return 0.5

    monkeypatch.setattr(
        "baybe.recommenders.pure.tfpr._auto_top_fraction", auto_top_fraction
    )

    actual = _tfpr_fitness(
        np.array([[4.0], [3.0], [2.0], [1.0]]),
        np.array([1]),
        np.array([0.0]),
        None,
    )

    assert_allclose(actual, [1.05 / (3 * 2.05), 0.0, 0.0, 0.0])


def test_tfpr_fitness_matches_dense_reference() -> None:
    """The vectorized implementation matches a dense reference."""
    values = np.array(
        [
            [4.0, 0.0],
            [3.0, 3.0],
            [2.0, 2.7],
            [1.0, 4.0],
        ]
    )
    weights = np.array([2, 1])
    tolerances = np.array([0.0, 0.2])

    actual = _tfpr_fitness(values, weights, tolerances, 0.75)
    expected = _dense_tfpr_fitness(values, weights, tolerances, 0.75)

    assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("n_candidates", "n_objectives", "top_fraction", "seed"),
    [
        pytest.param(2, 2, 1.0, 1, id="minimum"),
        pytest.param(7, 3, 0.4, 2, id="inactive_candidates"),
        pytest.param(20, 5, 0.2, 3, id="many_inactive_candidates"),
        pytest.param(25, 10, 0.7, 4, id="many_objectives"),
        pytest.param(7, 3, None, 5, id="automatic_fraction"),
    ],
)
def test_tfpr_fitness_matches_dense_reference_randomized(
    n_candidates: int,
    n_objectives: int,
    top_fraction: float | None,
    seed: int,
) -> None:
    """Vectorized fitness matches the dense reference across varied inputs."""
    rng = np.random.default_rng(seed)
    values = rng.integers(-5, 6, size=(n_candidates, n_objectives)).astype(float)
    weights = rng.integers(0, 5, size=n_objectives)
    weights[0] = 1
    tolerances = rng.choice([0.0, 0.05, 0.2], size=n_objectives)

    actual = _tfpr_fitness(values, weights, tolerances, top_fraction)
    expected = _dense_tfpr_fitness(values, weights, tolerances, top_fraction)

    assert_allclose(actual, expected)


def test_recommend_uses_optimism_and_filtered_candidates(
    searchspace: SearchSpace,
    pareto_objective: ParetoObjective,
    measurements: pd.DataFrame,
) -> None:
    """Campaign filtering is respected before optimistic TFPR ranking."""
    surrogate = FakePosteriorStatsSurrogate()
    campaign = Campaign(
        searchspace,
        pareto_objective,
        recommender=TFPRRecommender(
            surrogate_model=surrogate,
            weights={"yield": 2},
            optimism_lambda=1.0,
            top_fraction=1.0,
        ),
        allow_recommending_already_measured=False,
    )
    campaign.add_measurements(measurements)

    recommendation = campaign.recommend(batch_size=2)

    assert surrogate.candidates is not None
    assert surrogate.candidates["p"].tolist() == [0, 1, 2]
    assert recommendation["p"].tolist() == [1, 2]


def test_two_phase_campaign_switches_to_tfpr(
    searchspace: SearchSpace,
    pareto_objective: ParetoObjective,
) -> None:
    """The documented two-phase workflow switches from random sampling to TFPR."""
    surrogate = FakePosteriorStatsSurrogate()
    campaign = Campaign(
        searchspace,
        pareto_objective,
        recommender=TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=TFPRRecommender(
                surrogate_model=surrogate,
                weights={"yield": 0, "cost": 1},
                top_fraction=1.0,
            ),
        ),
        allow_recommending_already_measured=False,
    )
    initial = campaign.recommend(batch_size=1)
    measured = initial.assign(
        yield_=initial["p"].astype(float), cost=-initial["p"].astype(float)
    ).rename(columns={"yield_": "yield"})
    campaign.add_measurements(measured)

    recommendation = campaign.recommend(batch_size=1)

    assert surrogate.candidates is not None
    assert initial["p"].item() not in surrogate.candidates["p"].tolist()
    assert recommendation["p"].item() in surrogate.candidates["p"].tolist()


def test_campaign_get_surrogate_accepts_tfpr(
    searchspace: SearchSpace,
    pareto_objective: ParetoObjective,
    measurements: pd.DataFrame,
) -> None:
    """Campaign exposes the surrogate of TFPR recommenders."""
    surrogate = FakePosteriorStatsSurrogate()
    campaign = Campaign(
        searchspace,
        pareto_objective,
        recommender=TFPRRecommender(surrogate_model=surrogate),
    )
    campaign.add_measurements(measurements)

    assert campaign.get_surrogate() is surrogate
    assert surrogate.measurements is not None


def test_recommend_checks_surrogate_capability_before_fitting(
    searchspace: SearchSpace,
    pareto_objective: ParetoObjective,
    measurements: pd.DataFrame,
) -> None:
    """Unsupported surrogates are rejected before fitting."""
    surrogate = FakeUnsupportedSurrogate()
    recommender = TFPRRecommender(surrogate_model=surrogate)

    with pytest.raises(IncompatibilityError, match="posterior_stats"):
        recommender.recommend(1, searchspace, pareto_objective, measurements)

    assert not surrogate.was_fit


@pytest.mark.parametrize(
    ("recommender", "match"),
    [
        pytest.param(
            TFPRRecommender(weights={"unknown": 1}),
            "unknown targets",
            id="unknown_weight",
        ),
        pytest.param(
            TFPRRecommender(tolerances={"unknown": 0.1}),
            "unknown targets",
            id="unknown_tolerance",
        ),
        pytest.param(
            TFPRRecommender(weights={"yield": 0, "cost": 0}),
            "positive",
            id="all_zero_weights",
        ),
    ],
)
def test_target_mapping_validation(
    recommender: TFPRRecommender,
    match: str,
    searchspace: SearchSpace,
    pareto_objective: ParetoObjective,
    measurements: pd.DataFrame,
) -> None:
    """Target-specific mappings are validated against the objective."""
    with pytest.raises(ValueError, match=match):
        recommender.recommend(1, searchspace, pareto_objective, measurements)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        pytest.param({"weights": {"yield": -1}}, ValueError, "between", id="w_low"),
        pytest.param({"weights": {"yield": 11}}, ValueError, "between", id="w_high"),
        pytest.param({"weights": {1: 1}}, TypeError, "string", id="w_key"),
        pytest.param(
            {"tolerances": {"yield": -0.1}},
            ValueError,
            "nonnegative",
            id="tol_low",
        ),
        pytest.param(
            {"tolerances": {"yield": float("inf")}},
            ValueError,
            "finite",
            id="tol_inf",
        ),
        pytest.param(
            {"optimism_lambda": -1.0},
            ValueError,
            "nonnegative",
            id="lambda_low",
        ),
        pytest.param(
            {"optimism_lambda": float("nan")},
            ValueError,
            "finite",
            id="lambda_nan",
        ),
        pytest.param({"top_fraction": 0.0}, ValueError, "0 < f", id="top_zero"),
        pytest.param({"top_fraction": 1.1}, ValueError, "0 < f", id="top_high"),
    ],
)
def test_init_validation(
    kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    """Invalid TFPR constructor arguments are rejected."""
    with pytest.raises(error, match=match):
        TFPRRecommender(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "attribute", "expected"),
    [
        pytest.param(
            {"weights": {"yield": True}},
            "weights",
            {"yield": 1},
            id="boolean_weight",
        ),
        pytest.param(
            {"weights": {"yield": 1.9}},
            "weights",
            {"yield": 1},
            id="fractional_weight",
        ),
        pytest.param(
            {"tolerances": {"yield": True}},
            "tolerances",
            {"yield": 1.0},
            id="boolean_tolerance",
        ),
        pytest.param(
            {"optimism_lambda": True},
            "optimism_lambda",
            1.0,
            id="boolean_optimism",
        ),
        pytest.param(
            {"top_fraction": True},
            "top_fraction",
            1.0,
            id="boolean_top_fraction",
        ),
    ],
)
def test_init_numeric_conversion(
    kwargs: dict[str, Any], attribute: str, expected: Any
) -> None:
    """Numeric constructor inputs are converted to their declared field types."""
    recommender = TFPRRecommender(**kwargs)

    assert getattr(recommender, attribute) == expected


def test_recommend_honors_target_direction(
    searchspace: SearchSpace, measurements: pd.DataFrame
) -> None:
    """Minimization reverses the posterior mean before TFPR ranking."""
    surrogate = FakePosteriorStatsSurrogate()
    maximize_cost = ParetoObjective([NumericalTarget("yield"), NumericalTarget("cost")])
    minimize_cost = ParetoObjective(
        [NumericalTarget("yield"), NumericalTarget("cost", minimize=True)]
    )
    recommender = TFPRRecommender(
        surrogate_model=surrogate,
        weights={"yield": 0, "cost": 1},
        top_fraction=1.0,
    )

    maximize_rec = recommender.recommend(1, searchspace, maximize_cost, measurements)
    minimize_rec = recommender.recommend(1, searchspace, minimize_cost, measurements)

    assert maximize_rec["p"].item() == 0
    assert minimize_rec["p"].item() == 3


def test_small_top_fraction_still_produces_ranking_signal() -> None:
    """Small positive fractions retain enough candidates for a comparison."""
    values = np.array([[0.0], [1.0], [2.0], [3.0]])
    fitness = _tfpr_fitness(
        values,
        np.array([1]),
        np.array([0.0]),
        0.01,
    )

    assert fitness[3] > fitness[2]
    assert fitness[2] == fitness[1] == fitness[0] == 0.0


def test_recommend_rejects_target_transformations(
    searchspace: SearchSpace, measurements: pd.DataFrame
) -> None:
    """TFPR fails loudly when posterior values do not match objective values."""
    objective = ParetoObjective(
        [
            NumericalTarget.match_bell("yield", match_value=1.0, sigma=1.0),
            NumericalTarget("cost"),
        ]
    )

    with pytest.raises(IncompatibilityError, match="identity target transformations"):
        TFPRRecommender(surrogate_model=FakePosteriorStatsSurrogate()).recommend(
            1, searchspace, objective, measurements
        )


def test_default_gaussian_process_surrogate_recommends_candidate(
    searchspace: SearchSpace, pareto_objective: ParetoObjective
) -> None:
    """The default replicated Gaussian process supports TFPR end to end."""
    campaign = Campaign(
        searchspace,
        pareto_objective,
        recommender=TFPRRecommender(top_fraction=1.0),
        allow_recommending_already_measured=False,
    )
    campaign.add_measurements(
        pd.DataFrame(
            {
                "p": [0, 3],
                "yield": [0.0, 3.0],
                "cost": [3.0, 0.0],
            }
        )
    )

    recommendation = campaign.recommend(batch_size=1)

    assert len(recommendation) == 1
    assert recommendation["p"].item() in {1, 2}


def test_recommend_requires_pareto_objective(
    searchspace: SearchSpace, measurements: pd.DataFrame
) -> None:
    """TFPR rejects non-Pareto objectives."""
    objective = NumericalTarget("yield").to_objective()

    with pytest.raises(IncompatibilityError, match="ParetoObjective"):
        TFPRRecommender(surrogate_model=FakePosteriorStatsSurrogate()).recommend(
            1, searchspace, objective, measurements
        )


def test_recommend_rejects_continuous_searchspace(
    pareto_objective: ParetoObjective,
) -> None:
    """TFPR has discrete-only compatibility."""
    searchspace = NumericalContinuousParameter("p", (0, 1)).to_searchspace()
    measurements = pd.DataFrame({"p": [0.0], "yield": [0.0], "cost": [0.0]})

    with pytest.raises(IncompatibilityError, match="discrete"):
        TFPRRecommender(surrogate_model=FakePosteriorStatsSurrogate()).recommend(
            1, searchspace, pareto_objective, measurements
        )
