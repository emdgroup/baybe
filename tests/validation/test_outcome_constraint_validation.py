"""Validation tests for outcome constraints."""

import pandas as pd
import pytest
from pytest import param

from baybe.acquisition.acqfs import qUpperConfidenceBound
from baybe.constraints.outcome import OutcomeConstraint
from baybe.exceptions import IncompatibilityError, OutcomeConstraintIgnoredWarning
from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.targets.numerical import NumericalTarget

# ---------------------------------------------------------------------------
# Module-level test data
# ---------------------------------------------------------------------------

_temp_target = NumericalTarget("temperature", minimize=None)
_yield_target = NumericalTarget("yield", minimize=False)

_parameter = NumericalDiscreteParameter(name="x", values=list(range(1, 11)))
_searchspace = SearchSpace.from_product(parameters=[_parameter])
_measurements = pd.DataFrame(
    {
        "x": [1, 2, 3, 4, 5],
        "yield": [10.0, 20.0, 30.0, 40.0, 50.0],
        "temperature": [80.0, 90.0, 95.0, 105.0, 110.0],
    }
)

# ---------------------------------------------------------------------------
# OutcomeConstraint creation validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        param(
            {"target": "temperature", "operator": "<=", "threshold": 100.0},
            TypeError,
            None,
            id="non_target_object",
        ),
        param(
            {"target": _temp_target, "operator": ">", "threshold": 100.0},
            ValueError,
            "'operator' must be in",
            id="invalid_operator",
        ),
        param(
            {"target": _temp_target, "operator": "<=", "threshold": "high"},
            ValueError,
            "could not convert string to float",
            id="non_numeric_threshold",
        ),
    ],
)
def test_invalid_outcome_constraint(kwargs, error, match):
    """Invalid OutcomeConstraint arguments raise expected errors."""
    with pytest.raises(error, match=match):
        OutcomeConstraint(**kwargs)


# ---------------------------------------------------------------------------
# Objective + constraint combination validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("constraint_target", "match"),
    [
        param(
            _yield_target,
            "both optimized and constrained",
            id="same_target",
        ),
        param(
            NumericalTarget("yield", minimize=None),
            "both optimized and constrained",
            id="same_name_different_object",
        ),
        param(
            NumericalTarget("temperature", minimize=False),
            "must have minimize=None",
            id="minimize_not_none",
        ),
    ],
)
def test_invalid_objective_with_outcome_constraint(constraint_target, match):
    """Invalid outcome constraint + objective combinations raise expected errors."""
    constraint = OutcomeConstraint(constraint_target, "<=", 100.0)
    with pytest.raises(ValueError, match=match):
        SingleTargetObjective(target=_yield_target, outcome_constraints=(constraint,))


# ---------------------------------------------------------------------------
# Recommender incompatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("recommender", "expected"),
    [
        param(
            BotorchRecommender(acquisition_function=qUpperConfidenceBound()),
            IncompatibilityError,
            id="incompatible_acqf",
        ),
        param(
            RandomRecommender(),
            OutcomeConstraintIgnoredWarning,
            id="nonpredictive_recommender",
        ),
    ],
)
def test_recommender_constraint_incompatibility(recommender, expected):
    """Incompatible recommenders raise errors or emit warnings."""
    constraint = OutcomeConstraint(target=_temp_target, operator="<=", threshold=100.0)
    objective = SingleTargetObjective(
        target=_yield_target, outcome_constraints=(constraint,)
    )

    context = (
        pytest.warns(expected)
        if issubclass(expected, Warning)
        else pytest.raises(expected, match="does not support outcome")
    )
    with context:
        recommender.recommend(
            batch_size=2,
            searchspace=_searchspace,
            objective=objective,
            measurements=_measurements,
        )
