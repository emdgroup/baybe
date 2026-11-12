"""Tests for the experimental feasibility recommender."""

import numpy as np
import pandas as pd
import pytest
import torch
from pandas.testing import assert_frame_equal, assert_index_equal

from baybe.exceptions import InvalidTargetValueError
from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.recommenders.pure.experimental_feasibility import (
    ExperimentFeasibilityRecommender,
    InfeasibleMeasurementMode,
    _GaussianProcessClassifier,
    _GaussianProcessRegressor,
)
from baybe.searchspace import SearchSpace, SearchSpaceType
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.base import Surrogate
from baybe.targets import NumericalTarget
from baybe.targets.base import Target
from baybe.targets.binary import BinaryTarget
from tests.conftest import FAST_OPTIMIZATION_SETTINGS

_FEASIBILITY_TARGET = BinaryTarget(name="Feasibility")
"""The binary target encoding whether an experiment was feasible."""

_NUMERICAL_FEASIBILITY_TARGET = NumericalTarget(name="Constraint")
"""The numerical target encoding constraint values; non-positive values are feasible."""

_N_CONTINUOUS_SAMPLES = 4
"""Number of continuous configurations sampled per discrete configuration."""

_N_FREE_CONTINUOUS_SAMPLES = 32
"""Number of continuous configurations sampled in spaces without a discrete part."""

_RANDOM_SEED = 42
"""Seed making the generation of the fake measurements reproducible."""

_DUMMY_VALUE = -123.0
"""Dummy objective value used in the corresponding measurement mode."""

_PREDICTED_VALUE = 456.0
"""Predicted objective value returned by the stubbed surrogate."""

_DISCRETE_PARAMETER_NAMES = ["Categorical_1", "Categorical_2", "Num_disc_1"]
_CONTINUOUS_PARAMETER_NAMES = ["Conti_finite1", "Conti_finite2"]
_HYBRID_PARAMETER_NAMES = ["Categorical_1", "Num_disc_1", "Conti_finite1"]


def _make_recommender(
    target: Target = _FEASIBILITY_TARGET, **kwargs
) -> ExperimentFeasibilityRecommender:
    """Create the recommender under test."""
    return ExperimentFeasibilityRecommender(
        target=target,
        recommender=BotorchRecommender(**FAST_OPTIMIZATION_SETTINGS),
        **kwargs,
    )


def _make_measurements(
    searchspace: SearchSpace,
    targets,
    feasibility_target: Target = _FEASIBILITY_TARGET,
) -> pd.DataFrame:
    """Create deterministic measurements covering the whole search space.

    The feasibility values are derived from a single parameter, yielding an
    unambiguous decision boundary. Every discrete configuration is combined with
    continuous configurations from both sides of the boundary, so that the model
    cannot mistake a configuration for being entirely infeasible.
    """
    continuous = searchspace.continuous
    discrete = searchspace.discrete

    if discrete.parameters:
        discrete_part = discrete.exp_rep.reset_index(drop=True)
    else:
        discrete_part = pd.DataFrame([{}])

    if continuous.parameters:
        bounds = continuous.comp_rep_bounds.to_numpy()
        n_samples = (
            _N_CONTINUOUS_SAMPLES if discrete.parameters else _N_FREE_CONTINUOUS_SAMPLES
        )
        rng = np.random.default_rng(_RANDOM_SEED)
        continuous_part = pd.DataFrame(
            rng.uniform(bounds[0], bounds[1], size=(n_samples, bounds.shape[1])),
            columns=list(continuous.parameter_names),
        )
        split_parameter = continuous.parameters[0]
    else:
        continuous_part = pd.DataFrame([{}])
        split_parameter = next(p for p in discrete.parameters if p.is_numerical)

    measurements = discrete_part.merge(continuous_part, how="cross")
    values = measurements[split_parameter.name]
    midpoint = (values.min() + values.max()) / 2
    if isinstance(feasibility_target, BinaryTarget):
        measurements[feasibility_target.name] = values <= midpoint
    else:
        measurements[feasibility_target.name] = values - midpoint
    for target in targets:
        measurements[target.name] = np.linspace(0.0, 1.0, len(measurements))
    return measurements


def _stub_probabilities(
    monkeypatch, model_cls: type, column: int, threshold: float
) -> None:
    """Replace the model prediction with a deterministic threshold rule."""

    def posterior_probabilities(self, candidates_comp, /):
        return (candidates_comp[:, column] <= threshold).to(candidates_comp.dtype)

    monkeypatch.setattr(model_cls, "posterior_probabilities", posterior_probabilities)


@pytest.mark.parametrize(
    ("parameter_names", "searchspace_type"),
    [
        pytest.param(
            _DISCRETE_PARAMETER_NAMES, SearchSpaceType.DISCRETE, id="discrete"
        ),
        pytest.param(
            _CONTINUOUS_PARAMETER_NAMES, SearchSpaceType.CONTINUOUS, id="continuous"
        ),
        pytest.param(_HYBRID_PARAMETER_NAMES, SearchSpaceType.HYBRID, id="hybrid"),
    ],
)
def test_default_mode_recommendation(searchspace_type, targets, searchspace, objective):
    """The recommender returns a full batch for every search space type."""
    assert searchspace.type is searchspace_type

    measurements = _make_measurements(searchspace, targets)
    recommender = _make_recommender()

    rec = recommender.recommend(2, searchspace, objective, measurements)

    assert len(rec) == 2
    assert set(rec.columns) == set(searchspace.parameter_names)


def test_simple_model_fit():
    """The classifier learns a simple, clearly separable feasibility boundary."""
    parameter = NumericalDiscreteParameter(name="x", values=(0.0, 1.0, 2.0, 3.0))
    target = NumericalTarget(name="Target")
    objective = target.to_objective()
    searchspace = SearchSpace.from_product([parameter])

    measurements = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0] * 5, "Target": [0.0, 1.0, 2.0, 3.0] * 5}
    )
    measurements[_FEASIBILITY_TARGET.name] = measurements["x"] >= 2.0

    rec = _make_recommender().recommend(2, searchspace, objective, measurements)

    # The configurations below the learned boundary must be filtered out.
    assert set(rec["x"]) == {2.0, 3.0}


@pytest.mark.parametrize(
    "mode", list(InfeasibleMeasurementMode), ids=lambda mode: mode.name.lower()
)
def test_infeasible_measurement_modes(
    monkeypatch, mode, targets, searchspace, objective
):
    """Each measurement mode treats infeasible experiments as specified."""
    measurements = _make_measurements(searchspace, targets)
    target_name = targets[0].name

    def fake_posterior_stats(self, candidates, stats=("mean", "std")):
        return pd.DataFrame(
            {f"{target_name}_mean": np.full(len(candidates), _PREDICTED_VALUE)},
            index=candidates.index,
        )

    monkeypatch.setattr(
        BayesianRecommender,
        "get_surrogate",
        lambda *args, **kwargs: GaussianProcessSurrogate(),
    )
    monkeypatch.setattr(Surrogate, "posterior_stats", fake_posterior_stats)

    recommender = _make_recommender(
        infeasible_measurement_mode=mode,
        infeasible_dummy_value=_DUMMY_VALUE,
    )
    values = recommender._extract_values(measurements)
    is_feasible = recommender._observed_feasibility(values)
    is_infeasible = ~is_feasible.to_numpy()

    result = recommender._objective_measurements(
        measurements, is_feasible, objective, searchspace
    )

    if mode is InfeasibleMeasurementMode.OBSERVED:
        expected = measurements
    elif mode is InfeasibleMeasurementMode.EXCLUDE:
        expected = measurements.loc[~is_infeasible]
    else:
        expected = measurements.copy()
        imputation = (
            _DUMMY_VALUE
            if mode is InfeasibleMeasurementMode.DUMMY
            else _PREDICTED_VALUE
        )
        expected.loc[is_infeasible, target_name] = imputation

    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("target", "model_cls"),
    [
        pytest.param(_FEASIBILITY_TARGET, _GaussianProcessClassifier, id="binary"),
        pytest.param(
            _NUMERICAL_FEASIBILITY_TARGET, _GaussianProcessRegressor, id="numerical"
        ),
    ],
)
def test_discrete_candidates_are_filtered(monkeypatch, searchspace, target, model_cls):
    """Candidates predicted infeasible are removed from the discrete space."""
    assert searchspace.type is SearchSpaceType.DISCRETE
    split_name, split_value = "Num_disc_1", 1.5
    column = searchspace.comp_rep_columns.index(split_name)
    _stub_probabilities(monkeypatch, model_cls, column, split_value)

    recommender = _make_recommender(target=target)
    model = recommender._feasibility_model
    assert isinstance(model, model_cls)
    restricted = recommender._restrict_searchspace(searchspace, model)

    candidates_exp, _ = searchspace.discrete.get_candidates()
    retained_exp, _ = restricted.discrete.get_candidates()
    expected = candidates_exp.index[candidates_exp[split_name] <= split_value]

    assert_index_equal(retained_exp.index, expected)
    # The exclusion is only meaningful if some candidates were actually removed.
    assert (candidates_exp[split_name] > split_value).any()


@pytest.mark.parametrize(
    ("parameter_names", "searchspace_type"),
    [
        pytest.param(
            _CONTINUOUS_PARAMETER_NAMES, SearchSpaceType.CONTINUOUS, id="continuous"
        ),
        pytest.param(_HYBRID_PARAMETER_NAMES, SearchSpaceType.HYBRID, id="hybrid"),
    ],
)
@pytest.mark.parametrize(
    ("target", "model_cls"),
    [
        pytest.param(_FEASIBILITY_TARGET, _GaussianProcessClassifier, id="binary"),
        pytest.param(
            _NUMERICAL_FEASIBILITY_TARGET, _GaussianProcessRegressor, id="numerical"
        ),
    ],
)
def test_continuous_constraint_rejects_infeasible_configurations(
    monkeypatch, searchspace_type, searchspace, target, model_cls
):
    """The added nonlinear constraint rejects configurations predicted infeasible."""
    assert searchspace.type is searchspace_type
    split_name, split_value = "Conti_finite1", 0.5
    column = searchspace.comp_rep_columns.index(split_name)
    _stub_probabilities(monkeypatch, model_cls, column, split_value)

    recommender = _make_recommender(target=target)
    model = recommender._feasibility_model
    assert isinstance(model, model_cls)
    restricted = recommender._restrict_searchspace(searchspace, model)

    assert len(restricted.continuous.constraints_nonlin) == 1
    [(constraint, is_intra_point)] = restricted.continuous.constraints_nonlin[
        0
    ].to_botorch(restricted.continuous.parameters)
    assert is_intra_point

    def probe(value: float) -> torch.Tensor:
        point = torch.zeros(len(searchspace.comp_rep_columns), dtype=torch.double)
        point[column] = value
        return point

    assert constraint(probe(split_value / 2)) >= 0
    assert constraint(probe(1.5 * split_value)) < 0


@pytest.mark.parametrize(
    ("parameter_names", "searchspace_type"),
    [
        pytest.param(
            _DISCRETE_PARAMETER_NAMES, SearchSpaceType.DISCRETE, id="discrete"
        ),
        pytest.param(
            _CONTINUOUS_PARAMETER_NAMES, SearchSpaceType.CONTINUOUS, id="continuous"
        ),
        pytest.param(_HYBRID_PARAMETER_NAMES, SearchSpaceType.HYBRID, id="hybrid"),
    ],
)
def test_numerical_target_recommendation(
    searchspace_type, targets, searchspace, objective
):
    """A numerical feasibility target yields a full batch for every space type."""
    assert searchspace.type is searchspace_type

    measurements = _make_measurements(
        searchspace, targets, _NUMERICAL_FEASIBILITY_TARGET
    )
    recommender = _make_recommender(target=_NUMERICAL_FEASIBILITY_TARGET)

    rec = recommender.recommend(2, searchspace, objective, measurements)

    assert len(rec) == 2
    assert set(rec.columns) == set(searchspace.parameter_names)


def test_numerical_observed_feasibility():
    """Numerical constraint values are feasible when they are non-positive."""
    recommender = _make_recommender(target=_NUMERICAL_FEASIBILITY_TARGET)

    feasibility = recommender._observed_feasibility(pd.Series([-1.0, 0.0, 0.5]))

    assert feasibility.tolist() == [True, True, False]


def test_numerical_target_minimize_is_rejected():
    """A numerical feasibility target must not be configured for minimization."""
    with pytest.raises(ValueError, match="minimize"):
        _make_recommender(target=NumericalTarget(name="Constraint", minimize=True))


def test_non_numeric_constraint_values_raise(searchspace, targets):
    """Non-numeric measurement values cannot train a numerical feasibility model."""
    measurements = _make_measurements(
        searchspace, targets, _NUMERICAL_FEASIBILITY_TARGET
    )
    measurements[_NUMERICAL_FEASIBILITY_TARGET.name] = "invalid"
    recommender = _make_recommender(target=_NUMERICAL_FEASIBILITY_TARGET)

    with pytest.raises(InvalidTargetValueError, match="must contain numeric values"):
        recommender._extract_values(measurements)
