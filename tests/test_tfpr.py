"""Tests for the TFPR recommender."""

from __future__ import annotations

import math
from typing import Any, NoReturn

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from baybe import Campaign
from baybe.constraints import DiscreteBatchConstraint
from baybe.exceptions import IncompatibilityError, IncompatibleArgumentError
from baybe.objectives import DesirabilityObjective, ParetoObjective
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


class TablePosteriorStatsSurrogate:
    """Surrogate returning posterior statistics looked up via the parameter value."""

    def __init__(self, table: pd.DataFrame) -> None:
        self.table = table

    def fit(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        """Ignore the training data."""

    def posterior_stats(
        self,
        candidates: pd.DataFrame,
        stats: tuple[str, str] = ("mean", "std"),
    ) -> pd.DataFrame:
        """Return the table rows indexed by the candidates' parameter values."""
        assert stats == ("mean", "std")
        rows = self.table.loc[candidates["p"].astype(int)]
        return rows.set_axis(candidates.index)

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
    dominance_threshold: float = 0.5,
) -> np.ndarray:
    """Compute dense-matrix TFPR fitness for comparison in tests."""
    n_candidates, n_targets = values.shape
    counts = np.zeros((n_candidates, n_candidates), dtype=float)
    fraction = (
        _auto_top_fraction(n_candidates) if top_fraction is None else top_fraction
    )
    top_k = min(n_candidates, max(2, math.ceil(n_candidates * fraction)))

    for target_index in range(n_targets):
        weight = weights[target_index]
        if weight == 0:
            continue
        order = np.argsort(-values[:, target_index], kind="stable")[:top_k]
        for rank, left in enumerate(order):
            for right in order[rank + 1 :]:
                left_value = values[left, target_index]
                right_value = values[right, target_index]
                difference = abs(left_value - right_value)
                scale = max(abs(left_value), abs(right_value))
                tolerance = tolerances[target_index]
                if left_value == right_value or difference <= tolerance * scale:
                    counts[left, right] += weight / 2
                    counts[right, left] += weight / 2
                else:
                    counts[left, right] += weight

    total_weight = weights.sum()
    dominating_threshold = dominance_threshold * total_weight
    submitting_threshold = (1.0 - dominance_threshold) * total_weight
    fitness = np.zeros(n_candidates, dtype=float)
    for candidate_index in range(n_candidates):
        others = np.arange(n_candidates) != candidate_index
        mean_dominance = counts[candidate_index].sum() / (
            (n_candidates - 1) * total_weight
        )
        row = counts[candidate_index, others]
        n_dominating = np.count_nonzero(row > dominating_threshold)
        n_submitting = np.count_nonzero(row < submitting_threshold)
        fitness[candidate_index] = (
            mean_dominance * (n_dominating + _EPSILON) / (n_submitting + _EPSILON)
        )
    return fitness


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


_MAJORITY_VS_STRICT_VALUES = [[3.0, 3.0, 1.0], [2.0, 2.0, 3.0], [1.0, 1.0, 2.0]]
"""Candidates where the first wins a majority against all others and the second wins
all targets against the third."""


@pytest.mark.parametrize(
    ("values", "top_fraction", "dominance_threshold", "expected"),
    [
        pytest.param(
            [[3.0], [2.0], [1.0]],
            1.0,
            0.5,
            [41.0, 0.5, 0.0],
            id="known_ranking",
        ),
        pytest.param(
            [[1.0], [1.0]],
            1.0,
            0.5,
            [0.5, 0.5],
            id="exact_tie",
        ),
        pytest.param(
            [[4.0], [3.0], [2.0], [1.0]],
            0.5,
            0.5,
            [1.05 / (3 * 2.05), 0.0, 0.0, 0.0],
            id="top_fraction",
        ),
        pytest.param(
            _MAJORITY_VS_STRICT_VALUES,
            1.0,
            0.5,
            [
                2 / 3 * (2 + _EPSILON) / _EPSILON,
                2 / 3,
                1 / 6 * _EPSILON / (2 + _EPSILON),
            ],
            id="majority_threshold",
        ),
        pytest.param(
            _MAJORITY_VS_STRICT_VALUES,
            1.0,
            0.8,
            [
                2 / 3,
                2 / 3 * (1 + _EPSILON) / _EPSILON,
                1 / 6 * _EPSILON / (1 + _EPSILON),
            ],
            id="strict_threshold",
        ),
    ],
)
def test_tfpr_fitness_matches_hand_calculation(
    values: list[list[float]],
    top_fraction: float,
    dominance_threshold: float,
    expected: list[float],
) -> None:
    """TFPR fitness matches values calculated directly from its definition."""
    n_targets = len(values[0])
    actual = _tfpr_fitness(
        np.array(values),
        np.ones(n_targets),
        np.zeros(n_targets),
        top_fraction,
        dominance_threshold,
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
        np.array([1.0]),
        np.array([0.0]),
        None,
        0.5,
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
    weights = np.array([2.0, 1.0])
    tolerances = np.array([0.0, 0.2])

    actual = _tfpr_fitness(values, weights, tolerances, 0.75, 0.5)
    expected = _dense_tfpr_fitness(values, weights, tolerances, 0.75)

    assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "dominance_threshold",
    [
        pytest.param(0.5, id="majority"),
        pytest.param(0.8, id="strict"),
        pytest.param(0.99, id="near_unanimous"),
    ],
)
@pytest.mark.parametrize(
    ("n_candidates", "n_targets", "top_fraction", "seed"),
    [
        pytest.param(2, 2, 1.0, 1, id="minimum"),
        pytest.param(7, 3, 0.4, 2, id="inactive_candidates"),
        pytest.param(20, 5, 0.2, 3, id="many_inactive_candidates"),
        pytest.param(25, 10, 0.7, 4, id="many_targets"),
        pytest.param(7, 3, None, 5, id="automatic_fraction"),
    ],
)
def test_tfpr_fitness_matches_dense_reference_randomized(
    n_candidates: int,
    n_targets: int,
    top_fraction: float | None,
    seed: int,
    dominance_threshold: float,
) -> None:
    """Vectorized fitness matches the dense reference across varied inputs."""
    rng = np.random.default_rng(seed)
    values = rng.integers(-5, 6, size=(n_candidates, n_targets)).astype(float)
    # Quarter steps keep the weight sums exact, so the reference needs no margin
    weights = rng.integers(0, 20, size=n_targets) / 4
    weights[0] = 1.0
    tolerances = rng.choice([0.0, 0.05, 0.2], size=n_targets)

    actual = _tfpr_fitness(
        values, weights, tolerances, top_fraction, dominance_threshold
    )
    expected = _dense_tfpr_fitness(
        values, weights, tolerances, top_fraction, dominance_threshold
    )

    assert_allclose(actual, expected)


def test_tfpr_fitness_absorbs_weight_rounding_errors() -> None:
    """Secured weights equal to a threshold up to rounding count as reaching it."""
    # The first candidate wins the first two targets and the second candidate the
    # third one, so each secures exactly half of the total weight. In floating point,
    # however, 0.1 + 0.2 exceeds 0.3, and 0.3 falls short of half the summed weights.
    values = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    actual = _tfpr_fitness(values, np.array([0.1, 0.2, 0.3]), np.zeros(3), 1.0, 0.5)

    assert_allclose(actual, [0.5, 0.5])


@pytest.mark.parametrize("scale", [0.1, 3.0], ids=["scale_down", "scale_up"])
def test_tfpr_fitness_is_invariant_to_weight_scaling(scale: float) -> None:
    """Only the relative weights matter for the ranking."""
    rng = np.random.default_rng(0)
    values = rng.integers(-3, 4, size=(15, 4)).astype(float)
    weights = np.array([1.0, 2.0, 1.0, 3.0])

    reference = _tfpr_fitness(values, weights, np.zeros(4), 1.0, 0.5)
    scaled = _tfpr_fitness(values, scale * weights, np.zeros(4), 1.0, 0.5)

    assert_allclose(scaled, reference)


def test_small_top_fraction_still_produces_ranking_signal() -> None:
    """Small positive fractions retain enough candidates for a comparison."""
    values = np.array([[0.0], [1.0], [2.0], [3.0]])
    fitness = _tfpr_fitness(values, np.array([1.0]), np.array([0.0]), 0.01, 0.5)

    assert fitness[3] > fitness[2]
    assert fitness[2] == fitness[1] == fitness[0] == 0.0


def _objective(minimize_cost: bool = True) -> ParetoObjective:
    """Create a two-target Pareto objective."""
    return ParetoObjective(
        [NumericalTarget("yield"), NumericalTarget("cost", minimize=minimize_cost)]
    )


def _recommender(
    surrogate: Any | None = None,
    weights: dict[str, float] | None = None,
    tolerances: dict[str, float] | None = None,
    **kwargs: Any,
) -> TFPRRecommender:
    """Create a TFPR recommender with an optional custom surrogate and settings."""
    if surrogate is None:
        recommender = TFPRRecommender(**kwargs)
    else:
        recommender = TFPRRecommender(surrogate_model=surrogate, **kwargs)
    recommender._weights = weights
    recommender._tolerances = tolerances
    return recommender


def test_recommend_uses_optimism_and_filtered_candidates(
    searchspace: SearchSpace, measurements: pd.DataFrame
) -> None:
    """Campaign filtering is respected before optimistic TFPR ranking."""
    surrogate = FakePosteriorStatsSurrogate()
    campaign = Campaign(
        searchspace,
        _objective(),
        recommender=_recommender(
            surrogate,
            weights={"yield": 2, "cost": 1},
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


def test_two_phase_campaign_switches_to_tfpr(searchspace: SearchSpace) -> None:
    """The two-phase workflow switches from random sampling to TFPR ranking."""
    surrogate = FakePosteriorStatsSurrogate()
    campaign = Campaign(
        searchspace,
        _objective(),
        recommender=TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=_recommender(
                surrogate, weights={"yield": 0, "cost": 1}, top_fraction=1.0
            ),
        ),
        allow_recommending_already_measured=False,
    )
    initial = campaign.recommend(batch_size=1)
    p = initial["p"].astype(float)
    campaign.add_measurements(initial.assign(**{"yield": p, "cost": -p}))

    recommendation = campaign.recommend(batch_size=1)

    assert surrogate.candidates is not None
    assert initial["p"].item() not in surrogate.candidates["p"].tolist()
    assert recommendation["p"].item() in surrogate.candidates["p"].tolist()


def test_campaign_get_surrogate_with_tfpr(
    searchspace: SearchSpace, measurements: pd.DataFrame
) -> None:
    """Campaign exposes the surrogate of the TFPR recommender."""
    surrogate = FakePosteriorStatsSurrogate()
    campaign = Campaign(searchspace, _objective(), recommender=_recommender(surrogate))
    campaign.add_measurements(measurements)

    assert campaign.get_surrogate() is surrogate
    assert surrogate.measurements is not None


@pytest.mark.parametrize(
    ("weights", "tolerances"),
    [
        pytest.param({"yield": 1}, None, id="missing_weight"),
        pytest.param({"yield": 1, "cost": 1, "purity": 1}, None, id="unknown_weight"),
        pytest.param(None, {"cost": 0.1}, id="missing_tolerance"),
        pytest.param(
            None, {"yield": 0.1, "cost": 0.1, "purity": 0.1}, id="unknown_tolerance"
        ),
    ],
)
def test_recommend_rejects_settings_not_matching_targets(
    searchspace: SearchSpace,
    measurements: pd.DataFrame,
    weights: dict[str, float] | None,
    tolerances: dict[str, float] | None,
) -> None:
    """Private settings must cover exactly the objective targets, checked pre-fit."""
    surrogate = FakePosteriorStatsSurrogate()
    recommender = _recommender(surrogate, weights=weights, tolerances=tolerances)

    with pytest.raises(IncompatibilityError, match="exactly one value for each"):
        recommender.recommend(1, searchspace, _objective(), measurements)

    assert surrogate.measurements is None


def test_recommend_checks_surrogate_capability_before_fitting(
    searchspace: SearchSpace, measurements: pd.DataFrame
) -> None:
    """Surrogates without posterior statistics are rejected before fitting."""
    surrogate = FakeUnsupportedSurrogate()

    with pytest.raises(IncompatibilityError, match="posterior_stats"):
        _recommender(surrogate).recommend(1, searchspace, _objective(), measurements)

    assert not surrogate.was_fit


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"optimism_lambda": -1.0}, ">= 0", id="lambda_low"),
        pytest.param({"optimism_lambda": float("nan")}, "nan", id="lambda_nan"),
        pytest.param({"top_fraction": 0.0}, "> 0", id="top_zero"),
        pytest.param({"top_fraction": 1.1}, "<= 1", id="top_high"),
        pytest.param({"dominance_threshold": 0.4}, ">= 0.5", id="threshold_low"),
        pytest.param({"dominance_threshold": 1.0}, "< 1", id="threshold_high"),
        pytest.param({"dominance_threshold": float("nan")}, "nan", id="threshold_nan"),
    ],
)
def test_init_validation(kwargs: dict[str, Any], match: str) -> None:
    """Invalid TFPR recommender arguments are rejected."""
    with pytest.raises(ValueError, match=match):
        TFPRRecommender(**kwargs)


@pytest.mark.parametrize("attribute", ["_weights", "_tolerances"])
@pytest.mark.parametrize(
    ("value", "error", "match"),
    [
        pytest.param({"yield": -1.0}, ValueError, ">= 0", id="negative"),
        pytest.param({"yield": float("nan")}, ValueError, "nan", id="nan"),
        pytest.param({"yield": float("inf")}, ValueError, "inf", id="inf"),
        pytest.param({1: 1.0}, TypeError, "must be <class 'str'>", id="key"),
    ],
)
def test_private_settings_validation(
    attribute: str, value: Any, error: type[Exception], match: str
) -> None:
    """Invalid private settings are rejected upon assignment."""
    with pytest.raises(error, match=match):
        setattr(TFPRRecommender(), attribute, value)


def test_private_weights_require_positive_weight() -> None:
    """Weights that are all zero are rejected upon assignment."""
    with pytest.raises(ValueError, match="positive"):
        TFPRRecommender()._weights = {"yield": 0, "cost": 0.0}


@pytest.mark.parametrize(
    ("kwargs", "attribute", "expected"),
    [
        pytest.param({}, "optimism_lambda", 0.0, id="default_optimism"),
        pytest.param({}, "top_fraction", None, id="default_top_fraction"),
        pytest.param({}, "dominance_threshold", 0.5, id="default_threshold"),
        pytest.param({}, "_weights", None, id="default_weights"),
        pytest.param({}, "_tolerances", None, id="default_tolerances"),
        pytest.param(
            {"optimism_lambda": True}, "optimism_lambda", 1.0, id="boolean_optimism"
        ),
        pytest.param(
            {"top_fraction": True}, "top_fraction", 1.0, id="boolean_top_fraction"
        ),
    ],
)
def test_init_numeric_conversion(
    kwargs: dict[str, Any], attribute: str, expected: Any
) -> None:
    """Numeric inputs are converted to their declared field types."""
    assert getattr(TFPRRecommender(**kwargs), attribute) == expected


@pytest.mark.parametrize("attribute", ["_weights", "_tolerances"])
def test_private_settings_conversion(attribute: str) -> None:
    """Private settings are copied and their values converted to floats."""
    value = {"yield": True, "cost": 2}
    recommender = TFPRRecommender()
    setattr(recommender, attribute, value)

    converted = getattr(recommender, attribute)
    assert converted == {"yield": 1.0, "cost": 2.0}
    assert all(type(v) is float for v in converted.values())
    assert converted is not value


def test_unspecified_weights_are_uniform(
    searchspace: SearchSpace, measurements: pd.DataFrame
) -> None:
    """Without explicit weights, all targets are weighted equally."""
    kwargs = {"optimism_lambda": 1.0, "top_fraction": 1.0}
    default = _recommender(FakePosteriorStatsSurrogate(), **kwargs)
    uniform = _recommender(
        FakePosteriorStatsSurrogate(), weights={"yield": 2.5, "cost": 2.5}, **kwargs
    )
    skewed = _recommender(
        FakePosteriorStatsSurrogate(), weights={"yield": 2.0, "cost": 1.0}, **kwargs
    )

    def recommend(recommender: TFPRRecommender) -> list[int]:
        rec = recommender.recommend(3, searchspace, _objective(), measurements)
        return rec["p"].tolist()

    assert recommend(default) == recommend(uniform) != recommend(skewed)


@pytest.mark.parametrize(
    ("minimize_cost", "expected"),
    [
        pytest.param(False, 0, id="maximize"),
        pytest.param(True, 3, id="minimize"),
    ],
)
def test_recommend_honors_target_direction(
    searchspace: SearchSpace,
    measurements: pd.DataFrame,
    minimize_cost: bool,
    expected: int,
) -> None:
    """Minimization reverses the posterior mean before TFPR ranking."""
    recommender = _recommender(
        FakePosteriorStatsSurrogate(),
        weights={"yield": 0, "cost": 1},
        top_fraction=1.0,
    )
    objective = _objective(minimize_cost=minimize_cost)

    recommendation = recommender.recommend(1, searchspace, objective, measurements)

    assert recommendation["p"].item() == expected


def test_default_gaussian_process_surrogate_recommends_candidate(
    searchspace: SearchSpace,
) -> None:
    """The default surrogate supports TFPR end to end."""
    campaign = Campaign(
        searchspace,
        _objective(),
        recommender=TFPRRecommender(top_fraction=1.0),
        allow_recommending_already_measured=False,
    )
    campaign.add_measurements(
        pd.DataFrame({"p": [0, 3], "yield": [0.0, 3.0], "cost": [3.0, 0.0]})
    )

    recommendation = campaign.recommend(batch_size=1)

    assert len(recommendation) == 1
    assert recommendation["p"].item() in {1, 2}


@pytest.mark.parametrize(
    ("objective", "match"),
    [
        pytest.param(
            DesirabilityObjective(
                [
                    NumericalTarget.normalized_ramp("yield", cutoffs=(0, 1)),
                    NumericalTarget.normalized_ramp("cost", cutoffs=(0, 1)),
                ]
            ),
            ParetoObjective.__name__,
            id="non_pareto",
        ),
        pytest.param(
            ParetoObjective(
                [
                    NumericalTarget.match_bell("yield", match_value=1.0, sigma=1.0),
                    NumericalTarget("cost"),
                ]
            ),
            "without transformations",
            id="transformed_target",
        ),
    ],
)
def test_recommend_rejects_incompatible_objectives(
    searchspace: SearchSpace,
    measurements: pd.DataFrame,
    objective: Objective,
    match: str,
) -> None:
    """TFPR requires a Pareto objective with untransformed targets."""
    with pytest.raises(IncompatibilityError, match=match):
        _recommender(FakePosteriorStatsSurrogate()).recommend(
            1, searchspace, objective, measurements
        )


def test_recommend_rejects_continuous_searchspace() -> None:
    """TFPR requires a discrete search space."""
    searchspace = NumericalContinuousParameter("p", (0, 1)).to_searchspace()
    measurements = pd.DataFrame({"p": [0.0], "yield": [0.0], "cost": [0.0]})

    with pytest.raises(IncompatibilityError, match="discrete"):
        _recommender(FakePosteriorStatsSurrogate()).recommend(
            1, searchspace, _objective(), measurements
        )


def test_recommend_rejects_subset_generating_constraints(
    measurements: pd.DataFrame,
) -> None:
    """TFPR does not support subset-generating constraints."""
    searchspace = SearchSpace.from_product(
        [
            NumericalDiscreteParameter("p", [0, 1, 2, 3]),
            NumericalDiscreteParameter("q", [0, 1]),
        ],
        [DiscreteBatchConstraint(parameters=["q"])],
    )
    measurements = measurements.assign(q=0)

    with pytest.raises(IncompatibilityError, match="subset-generating"):
        _recommender(FakePosteriorStatsSurrogate()).recommend(
            1, searchspace, _objective(), measurements
        )


def test_recommend_rejects_pending_experiments(
    searchspace: SearchSpace, measurements: pd.DataFrame
) -> None:
    """TFPR cannot use pending experiments and raises instead of ignoring them."""
    pending = measurements[["p"]]

    with pytest.raises(IncompatibleArgumentError, match="Pending experiments"):
        _recommender(FakePosteriorStatsSurrogate()).recommend(
            1, searchspace, _objective(), measurements, pending
        )


@pytest.mark.parametrize(
    ("dominance_threshold", "expected"),
    [
        pytest.param(0.5, 0, id="majority"),
        pytest.param(0.8, 1, id="strict"),
    ],
)
def test_recommend_applies_dominance_threshold(
    dominance_threshold: float, expected: int
) -> None:
    """Stricter thresholds stop rewarding majority wins as dominance."""
    names = ["a", "b", "c"]
    # Candidate 0 wins a majority against all others, candidate 1 wins all targets
    # against candidate 2
    means = pd.DataFrame(_MAJORITY_VS_STRICT_VALUES, columns=names)
    surrogate = TablePosteriorStatsSurrogate(
        means.add_suffix("_mean").assign(**{f"{n}_std": 0.0 for n in names})
    )
    recommender = _recommender(
        surrogate, top_fraction=1.0, dominance_threshold=dominance_threshold
    )
    searchspace = NumericalDiscreteParameter("p", [0, 1, 2]).to_searchspace()
    objective = ParetoObjective([NumericalTarget(n) for n in names])
    measurements = pd.DataFrame({"p": [0], "a": [0.0], "b": [0.0], "c": [0.0]})

    recommendation = recommender.recommend(1, searchspace, objective, measurements)

    assert recommendation["p"].item() == expected


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        pytest.param("yield_mean", np.nan, "finite", id="nan_mean"),
        pytest.param("yield_std", np.inf, "finite", id="inf_std"),
        pytest.param("yield_std", -1.0, "non-negative", id="negative_std"),
    ],
)
def test_recommend_rejects_invalid_posterior_stats(
    searchspace: SearchSpace,
    measurements: pd.DataFrame,
    column: str,
    value: float,
    match: str,
) -> None:
    """Invalid posterior statistics are rejected."""
    stats = pd.DataFrame(
        {"yield_mean": [1.0, 2.0, 3.0, 4.0], "yield_std": 0.0}
        | {"cost_mean": [1.0, 2.0, 3.0, 4.0], "cost_std": 0.0}
    )
    stats.loc[0, column] = value
    recommender = _recommender(TablePosteriorStatsSurrogate(stats))

    with pytest.raises(ValueError, match=match):
        recommender.recommend(1, searchspace, _objective(), measurements)
