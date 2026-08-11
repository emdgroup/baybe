"""Tests for ContinuousOptimizer."""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch

from baybe.constraints.continuous import ContinuousLinearConstraint
from baybe.exceptions import IncompatibilityError
from baybe.optimizers.continuous import ContinuousOptimizer
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.searchspace import SubspaceContinuous

_PC1 = NumericalContinuousParameter("x1", bounds=(0, 1))
_PC2 = NumericalContinuousParameter("x2", bounds=(-1, 0))

_PATCH_TARGET = "botorch.optim.optimize_acqf"
_MOCK_PTS = torch.zeros(1, 2)
_MOCK_SCORES = torch.zeros(1)
_MOCK_RESULT = (_MOCK_PTS, _MOCK_SCORES)

_SS = SubspaceContinuous.from_product([_PC1, _PC2])
_SS_EQ = SubspaceContinuous.from_product(
    [_PC1, _PC2],
    constraints=[
        ContinuousLinearConstraint(
            parameters=["x1", "x2"], coefficients=[1.0, 1.0], operator="=", rhs=0.3
        )
    ],
)
_SS_INEQ = SubspaceContinuous.from_product(
    [_PC1, _PC2],
    constraints=[
        ContinuousLinearConstraint(
            parameters=["x1", "x2"], coefficients=[1.0, 1.0], operator=">=", rhs=0.3
        )
    ],
)
_SS_INTERPOINT = SubspaceContinuous.from_product(
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


@pytest.fixture(name="mock_score_fn")
def fixture_mock_score_fn() -> MagicMock:
    return MagicMock()


@pytest.mark.parametrize(
    ("sequential", "searchspace", "expected", "error"),
    [
        pytest.param("auto", _SS, True, None, id="auto_no_ip"),
        pytest.param("auto", _SS_INTERPOINT, False, None, id="auto_ip"),
        pytest.param(True, _SS, True, None, id="true_no_ip"),
        pytest.param(True, _SS_INTERPOINT, None, IncompatibilityError, id="true_ip"),
        pytest.param(False, _SS, False, None, id="false_no_ip"),
        pytest.param(False, _SS_INTERPOINT, False, None, id="false_ip"),
    ],
)
@patch(_PATCH_TARGET, return_value=_MOCK_RESULT)
def test_sequential_flag(
    mock_acqf, sequential, searchspace, expected, error, mock_score_fn
):
    """The sequential flag is resolved and forwarded correctly."""
    with pytest.raises(error, match="sequential") if error else nullcontext():
        ContinuousOptimizer(sequential=sequential)(1, mock_score_fn, searchspace)
    if error is None:
        assert mock_acqf.call_args.kwargs["sequential"] is expected


@pytest.mark.parametrize(
    ("opt_kwargs", "batch_size", "searchspace", "kwarg", "expected"),
    [
        pytest.param({}, 1, _SS, "fixed_features", None, id="fixed_features_none"),
        pytest.param(
            {}, 1, _SS, "equality_constraints", None, id="eq_constraints_none"
        ),
        pytest.param(
            {}, 1, _SS, "inequality_constraints", None, id="ineq_constraints_none"
        ),
        pytest.param(
            {}, 1, _SS_EQ, "equality_constraints", True, id="eq_constraints_set"
        ),
        pytest.param(
            {}, 1, _SS_INEQ, "inequality_constraints", True, id="ineq_constraints_set"
        ),
        pytest.param({"n_starts": 5}, 1, _SS, "num_restarts", 5, id="n_starts"),
        pytest.param(
            {"n_initial_samples": 32}, 1, _SS, "raw_samples", 32, id="n_initial_samples"
        ),
        pytest.param({}, 3, _SS, "q", 3, id="batch_size"),
        pytest.param({}, 1, _SS, "bounds", (2, 2), id="bounds_shape"),
    ],
)
@patch(_PATCH_TARGET, return_value=(torch.zeros(3, 2), torch.zeros(3)))
def test_optimizer_arguments(
    mock_acqf, opt_kwargs, batch_size, searchspace, kwarg, expected, mock_score_fn
):
    """Optimizer arguments are forwarded correctly to ``optimize_acqf``."""
    ContinuousOptimizer(**opt_kwargs)(batch_size, mock_score_fn, searchspace)
    value = mock_acqf.call_args.kwargs[kwarg]
    if kwarg == "bounds":
        assert tuple(value.shape) == expected
    elif isinstance(expected, bool):
        assert (value is not None) is expected
    else:
        assert value == expected


@patch(_PATCH_TARGET, return_value=_MOCK_RESULT)
def test_return_value(mock_acqf, mock_score_fn):
    """The optimizer returns exactly what ``optimize_acqf`` returned."""
    result_pts, result_scores = ContinuousOptimizer()(1, mock_score_fn, _SS)
    assert result_pts is _MOCK_PTS
    assert result_scores is _MOCK_SCORES
