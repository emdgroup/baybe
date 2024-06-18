"""Validation tests for acquisition functions."""

from contextlib import nullcontext

import pytest
from attr import NOTHING as N
from pytest import param

from baybe.acquisition import UCB, qLogNEI, qNEI, qNIPV, qUCB


@pytest.mark.parametrize(
    ("sampling_n_points", "sampling_fraction", "sampling_method", "error", "match"),
    [
        param(N, N, N, None, None, id="defaults"),
        param(-1, N, N, ValueError, "must be > 0", id="points_negative"),
        param("a", N, N, TypeError, "must be <class 'int'>", id="points_wrong_type"),
        param(N, 0.0, N, ValueError, "must be > 0.0", id="fraction_too_low"),
        param(N, 1.1, N, ValueError, "must be <= 1.0", id="fraction_too_high"),
        param(N, "abd", N, ValueError, "could not convert", id="fraction_wrong_type"),
        param(
            1, 1.0, N, ValueError, "specified at the same time", id="both_quantities"
        ),
        param(N, N, "method", ValueError, "not a valid", id="method_wrong_type"),
    ],
)
def test_qNIPV(sampling_n_points, sampling_fraction, sampling_method, error, match):
    """Invalid argument combinations raise an appropriate error."""
    kwargs = {
        k: v
        for k, v in {
            "sampling_n_points": sampling_n_points,
            "sampling_fraction": sampling_fraction,
            "sampling_method": sampling_method,
        }.items()
        if v is not N
    }
    with pytest.raises(error, match=match) if error is not None else nullcontext():
        qNIPV(**kwargs)


@pytest.mark.parametrize("acqf", [qNEI, qLogNEI])
def test_EI(acqf):
    """Providing a non-boolean argument to `prune_baseline` raises an error."""
    with pytest.raises(TypeError):
        acqf(1)


@pytest.mark.parametrize("acqf", [UCB, qUCB])
@pytest.mark.parametrize(
    ("beta", "match"),
    [
        param(-1.0, "must be >= 0.0", id="negative"),
        param("a", "could not convert", id="not_a_float"),
    ],
)
def test_UCB(acqf, beta, match):
    """Providing an invalid `beta` value raises an error."""
    with pytest.raises(ValueError, match=match):
        acqf(beta)
