"""Validation tests for BlockCoordinateOptimizer and CyclicOptimizationSchedule."""

from unittest.mock import MagicMock

import pytest
from botorch.acquisition import AnalyticAcquisitionFunction
from pytest import param

from baybe.constraints.continuous import ContinuousLinearConstraint
from baybe.constraints.discrete import DiscreteBatchConstraint
from baybe.exceptions import (
    IncompatibleAcquisitionFunctionError,
    IncompatibleSearchSpaceError,
)
from baybe.optimizers.composite import (
    BlockCoordinateOptimizer,
    CyclicOptimizationSchedule,
    OptimizationStep,
)
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace import SearchSpace

_PC1 = NumericalContinuousParameter("x1", bounds=(0, 1))
_PC2 = NumericalContinuousParameter("x2", bounds=(-1, 0))
_PD = NumericalDiscreteParameter("d", values=[1.0, 2.0, 3.0])

_SS = SearchSpace.from_product([_PC1, _PC2])
_SS_INTERPOINT = SearchSpace.from_product(
    [_PC1, _PC2],
    constraints=[
        ContinuousLinearConstraint(
            parameters=["x1"],
            coefficients=[1.0],
            operator="=",
            rhs=0.3,
            interpoint=True,
        )
    ],
)
_SS_DISCRETE_BATCH = SearchSpace.from_product(
    [_PD],
    constraints=[DiscreteBatchConstraint(parameters=["d"])],
)

_MOCK_OPTIMIZER = MagicMock()
_MOCK_STEP = OptimizationStep(selector="", optimizer=_MOCK_OPTIMIZER)
_MOCK_BCO = BlockCoordinateOptimizer(
    schedule=CyclicOptimizationSchedule(steps=(_MOCK_STEP,))
)
_MOCK_BCO_NOMATCH = BlockCoordinateOptimizer(
    schedule=CyclicOptimizationSchedule(
        steps=(OptimizationStep(selector="nomatch", optimizer=_MOCK_OPTIMIZER),)
    )
)


@pytest.mark.parametrize(
    ("steps", "error", "match"),
    [
        param((), ValueError, "Length of 'steps' must be >= 1", id="empty"),
        param(
            ("not_a_step",),
            TypeError,
            "must be <class '.*OptimizationStep'>",
            id="wrong_type",
        ),
    ],
)
def test_steps_invalid(steps, error, match):
    """Invalid ``steps`` values raise an error."""
    with pytest.raises(error, match=match):
        CyclicOptimizationSchedule(steps=steps)


@pytest.mark.parametrize(
    ("n_cycles", "error", "match"),
    [
        param(1.5, TypeError, "must be <class 'int'>", id="float"),
        param("2", TypeError, "must be <class 'int'>", id="string"),
        param(0, ValueError, "must be > 0", id="zero"),
        param(-1, ValueError, "must be > 0", id="negative"),
    ],
)
def test_n_cycles(n_cycles, error, match):
    """Invalid ``n_cycles`` values raise an error."""
    with pytest.raises(error, match=match):
        CyclicOptimizationSchedule(steps=(_MOCK_STEP,), n_cycles=n_cycles)


@pytest.mark.parametrize(
    ("bco", "score_fn", "searchspace", "error", "match"),
    [
        param(
            _MOCK_BCO,
            MagicMock(),
            _SS_INTERPOINT,
            IncompatibleSearchSpaceError,
            "interpoint",
            id="interpoint_constraints",
        ),
        param(
            _MOCK_BCO,
            MagicMock(),
            _SS_DISCRETE_BATCH,
            IncompatibleSearchSpaceError,
            "discrete batch",
            id="discrete_batch_constraint",
        ),
        param(
            _MOCK_BCO,
            MagicMock(spec=AnalyticAcquisitionFunction),
            _SS,
            IncompatibleAcquisitionFunctionError,
            "analytic",
            id="analytic_acqf",
        ),
        param(
            _MOCK_BCO_NOMATCH,
            MagicMock(),
            _SS,
            IncompatibleSearchSpaceError,
            "none of the specified steps",
            id="all_steps_skipped",
        ),
    ],
)
def test_raises_on_incompatible_call(bco, score_fn, searchspace, error, match):
    """Calling the optimizer with incompatible arguments raises an error."""
    with pytest.raises(error, match=match):
        bco(2, score_fn, searchspace)
