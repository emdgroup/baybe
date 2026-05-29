"""Tests for outcome constraints functionality."""

import numpy as np
import pandas as pd
import pytest
import torch
from pytest import param

from baybe.acquisition.acqfs import qLogExpectedImprovement
from baybe.constraints.outcome import OutcomeConstraint
from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.pareto import ParetoObjective
from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.targets.numerical import NumericalTarget
from baybe.transformations.basic import AffineTransformation

# ---------------------------------------------------------------------------
# Module-level test data
# ---------------------------------------------------------------------------

_temp_target = NumericalTarget("temperature", minimize=None)
_yield_target = NumericalTarget("yield", minimize=False)
_pressure_target = NumericalTarget("pressure", minimize=None)
_pressure_constraint = OutcomeConstraint(_pressure_target, ">=", 2.0)
_temp_constraint_le = OutcomeConstraint(_temp_target, "<=", 60.0)
_temp_constraint_ge = OutcomeConstraint(_temp_target, ">=", 100.0)

# Shared training data: yield and temperature both increase with x
_rng = np.random.default_rng(42)
_xs = np.linspace(1, 20, 15)
_noise = _rng.normal(0, 1, 15)
_measurements_temp = pd.DataFrame(
    {"x": _xs, "yield": 5 * _xs + _noise, "temperature": 6 * _xs + 20 + _noise}
)
# For >=: yield decreases with x, temperature increases → constraint pushes high
_measurements_temp_ge = pd.DataFrame(
    {"x": _xs, "yield": -5 * _xs + 100 + _noise, "temperature": 6 * _xs + 20 + _noise}
)

# Targets for multi-target objectives
_yield_target_normalized = NumericalTarget.normalized_ramp("yield", cutoffs=(0, 120))
_purity_target_normalized = NumericalTarget.normalized_ramp("purity", cutoffs=(0, 1))
_yield_target_pareto = NumericalTarget("yield", minimize=False)
_purity_target_pareto = NumericalTarget("purity", minimize=False)

# Measurements including yield, purity, and temperature for multi-target tests
_measurements_multi = pd.DataFrame(
    {
        "x": _xs,
        "yield": 5 * _xs + _noise,
        "purity": 0.04 * _xs + 0.2 + _noise * 0.01,
        "temperature": 6 * _xs + 20 + _noise,
    }
)

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operator", "sample_val", "threshold", "feasible"),
    [
        param("<=", 0.7, 150.0, True, id="le_feasible"),
        param("<=", 0.8, 150.0, False, id="le_infeasible"),
        param(">=", 0.8, 110.0, True, id="ge_feasible"),
        param(">=", 0.4, 110.0, False, id="ge_infeasible"),
    ],
)
def test_outcome_constraint(operator, sample_val, threshold, feasible):
    """Threshold maps to computational space; feasible iff constraint_value <= 0."""
    target = NumericalTarget(
        "temperature",
        transformation=AffineTransformation(factor=1 / 180, shift=-20 / 180),
        minimize=None,
    )
    constraint = OutcomeConstraint(target, operator, threshold)

    # Threshold transformation: experimental -> computational space
    expected_comp = (threshold - 20) / 180
    assert constraint.get_computational_threshold() == pytest.approx(
        expected_comp, abs=1e-5
    )

    # Operator semantics: feasible iff constraint_value <= 0
    constraint_func = constraint.to_botorch_constraint_func(0)
    result = constraint_func(torch.tensor([[sample_val]]))
    assert (result.item() <= 0) == feasible


# ---------------------------------------------------------------------------
# Objective with/without outcome constraint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outcome_constraints", "n_constraints", "n_targets", "has_str"),
    [
        param((), 0, 1, False, id="no_constraints"),
        param((_temp_constraint_le,), 1, 2, True, id="single_constraint"),
        param(
            (_temp_constraint_le, _pressure_constraint),
            2,
            3,
            True,
            id="multiple_constraints",
        ),
    ],
)
def test_objective_with_outcome_constraints(
    outcome_constraints, n_constraints, n_targets, has_str
):
    """SingleTargetObjective correctly integrates outcome constraints."""
    objective = SingleTargetObjective(
        target=_yield_target, outcome_constraints=outcome_constraints
    )

    assert len(objective.outcome_constraints) == n_constraints
    assert len(objective.to_botorch_constraints()) == n_constraints
    assert len(objective.targets) == n_targets
    assert len(objective.constraint_targets) == n_targets - 1
    assert objective._optimization_targets == (_yield_target,)
    assert ("Outcome Constraints" in str(objective)) == has_str


# ---------------------------------------------------------------------------
# Behavioral tests
# ---------------------------------------------------------------------------
@pytest.fixture(name="wide_searchspace")
def fixture_wide_searchspace():
    """A discrete search space with x in [1, 20]."""
    parameter = NumericalDiscreteParameter(name="x", values=list(range(1, 21)))
    return SearchSpace.from_product(parameters=[parameter])


@pytest.mark.parametrize(
    (
        "constrained_obj",
        "unconstrained_obj",
        "measurements",
        "assert_direction",
        "n_outputs",
    ),
    [
        param(
            SingleTargetObjective(
                target=_yield_target,
                outcome_constraints=(_temp_constraint_le,),
            ),
            SingleTargetObjective(target=_yield_target),
            _measurements_temp,
            "low",
            2,
            id="single_target_le",
        ),
        param(
            SingleTargetObjective(
                target=_yield_target,
                outcome_constraints=(_temp_constraint_ge,),
            ),
            SingleTargetObjective(target=_yield_target),
            _measurements_temp_ge,
            "high",
            2,
            id="single_target_ge",
        ),
        param(
            DesirabilityObjective(
                targets=[_yield_target_normalized, _purity_target_normalized],
                outcome_constraints=(_temp_constraint_le,),
            ),
            DesirabilityObjective(
                targets=[_yield_target_normalized, _purity_target_normalized],
            ),
            _measurements_multi,
            "low",
            3,
            id="desirability_le",
        ),
        param(
            ParetoObjective(
                targets=[_yield_target_pareto, _purity_target_pareto],
                outcome_constraints=(_temp_constraint_le,),
            ),
            ParetoObjective(
                targets=[_yield_target_pareto, _purity_target_pareto],
            ),
            _measurements_multi,
            "low",
            3,
            id="pareto_le",
        ),
    ],
)
def test_constraint_effectiveness(
    wide_searchspace,
    constrained_obj,
    unconstrained_obj,
    measurements,
    assert_direction,
    n_outputs,
):
    """Outcome constraints steer recommendations toward the feasible region.

    Compares constrained vs unconstrained recommendations to verify that
    constraints shift the mean recommendation toward the feasible region.
    """
    torch.manual_seed(42)

    means = []
    recommenders = []
    unconstrained_cols = [c for c in measurements.columns if c != "temperature"]
    for objective, train_data in zip(
        [constrained_obj, unconstrained_obj],
        [measurements, measurements[unconstrained_cols]],
    ):
        recommender = BotorchRecommender()
        result = recommender.recommend(
            batch_size=5,
            searchspace=wide_searchspace,
            objective=objective,
            measurements=train_data,
        )
        means.append(result["x"].mean())
        recommenders.append(recommender)

    mean_constrained, mean_unconstrained = means

    # Constraint steers recommendations toward the feasible region
    mean_small, mean_large = (
        (mean_constrained, mean_unconstrained)
        if assert_direction == "low"
        else (mean_unconstrained, mean_constrained)
    )
    assert mean_small < mean_large, (
        f"Constraint should steer toward {assert_direction} x: "
        f"constrained mean={mean_constrained}, "
        f"unconstrained mean={mean_unconstrained}"
    )

    # Constrained surrogate predicts all modeled targets
    posterior = recommenders[0]._surrogate_model.posterior(
        pd.DataFrame({"x": [3.0, 10.0, 17.0]}), joint=False
    )
    assert posterior.mean.shape[-1] == n_outputs, (
        f"Surrogate should predict {n_outputs} outputs, got {posterior.mean.shape[-1]}"
    )


@pytest.mark.parametrize(
    ("measurements", "best_f_range"),
    [
        pytest.param(
            _measurements_temp,
            (50.0, 80.0),
            id="mixed_feasibility",
        ),
        pytest.param(
            # All infeasible: falls back to global max from surrogate posterior mean
            _measurements_temp.assign(
                temperature=_measurements_temp["temperature"] + 200
            ),
            (80.0, 110.0),
            id="all_infeasible",
        ),
    ],
)
def test_best_f_feasibility(wide_searchspace, measurements, best_f_range):
    """best_f reflects best feasible value or falls back to global max."""
    torch.manual_seed(42)

    constraint = OutcomeConstraint(target=_temp_target, operator="<=", threshold=100.0)
    objective = SingleTargetObjective(
        target=_yield_target, outcome_constraints=(constraint,)
    )

    recommender = BotorchRecommender(acquisition_function=qLogExpectedImprovement())
    recommender.get_acquisition_function(
        searchspace=wide_searchspace,
        objective=objective,
        measurements=measurements,
    )

    best_f = recommender._botorch_acqf.best_f.item()
    assert best_f_range[0] < best_f < best_f_range[1], (
        f"best_f ({best_f}) should be in range {best_f_range}"
    )
