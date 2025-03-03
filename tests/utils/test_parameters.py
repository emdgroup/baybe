"""Tests for parameter utilities."""

import pytest
from pytest import param

from baybe.parameters import NumericalContinuousParameter
from baybe.parameters.numerical import _FixedNumericalContinuousParameter
from baybe.parameters.utils import activate_parameter
from baybe.utils.interval import Interval


def mirror_interval(interval: Interval) -> Interval:
    """Return an interval copy mirrored around the origin."""
    return Interval(lower=-interval.upper, upper=-interval.lower)


@pytest.mark.parametrize("mirror", [False, True], ids=["regular", "mirrored"])
@pytest.mark.parametrize(
    (
        "bounds",
        "thresholds",
        "expected_bounds",
        "error_message",
    ),
    [
        # Depending on whether a threshold lies on zero or not, the inactive range
        # can be a half-closed or open interval. To capture all possible scenarios, we
        # consider both with_zero/nonzero_threshold cases when necessary.
        param(
            Interval(lower=-1.0, upper=1.0),
            Interval(lower=-1.0, upper=1.0),
            Interval(lower=-1.0, upper=1.0),
            "",
            id="bounds_on_thresholds_with_nonzero_threshold",
        ),
        param(
            Interval(lower=0.0, upper=1.0),
            Interval(lower=0.0, upper=1.0),
            Interval(lower=1.0, upper=1.0),
            "",
            id="bounds_on_thresholds_with_zero_threshold",
        ),
        param(
            Interval(lower=-1.0, upper=1.0),
            Interval(lower=-1.5, upper=1.5),
            None,
            "cannot be set active",
            id="bounds_in_thresholds",
        ),
        param(
            Interval(lower=-1.0, upper=1.0),
            Interval(lower=-1.5, upper=1.0),
            Interval(lower=1.0, upper=1.0),
            "",
            id="bounds_in_thresholds_one_side_match_with_nonzero_threshold",
        ),
        param(
            Interval(lower=-1.0, upper=0.0),
            Interval(lower=-1.5, upper=0.0),
            None,
            "cannot be set active",
            id="bounds_in_thresholds_one_side_match_with_zero_threshold",
        ),
        param(
            Interval(lower=-1.0, upper=1.0),
            Interval(lower=-0.5, upper=0.5),
            Interval(lower=-1.0, upper=1.0),
            "",
            id="thresholds_in_bounds",
        ),
        param(
            Interval(lower=-1.0, upper=1.0),
            Interval(lower=-0.5, upper=1.0),
            Interval(lower=-1.0, upper=1.0),
            "",
            id="thresholds_in_bounds_one_side_match_with_nonzero_threshold",
        ),
        param(
            Interval(lower=-1.0, upper=0.0),
            Interval(lower=-0.5, upper=0.0),
            Interval(lower=-1.0, upper=-0.5),
            "",
            id="thresholds_in_bounds_one_side_match_with_zero_threshold",
        ),
        param(
            Interval(lower=-0.5, upper=1.0),
            Interval(lower=-1.0, upper=0.5),
            Interval(lower=0.5, upper=1.0),
            "",
            id="bounds_intersected_with_thresholds",
        ),
        param(
            Interval(lower=0.5, upper=1.0),
            Interval(lower=-1.0, upper=0.5),
            Interval(lower=0.5, upper=1.0),
            "",
            id="bounds_intersected_with_thresholds_on_nonzero_one_point",
        ),
        param(
            Interval(lower=0.0, upper=1.0),
            Interval(lower=-1.0, upper=0.0),
            Interval(lower=0.0, upper=1.0),
            "",
            id="bounds_intersected_with_thresholds_on_zero_one_point",
        ),
        param(
            Interval(lower=0.5, upper=1.0),
            Interval(lower=-1.0, upper=0.0),
            Interval(lower=0.5, upper=1.0),
            "",
            id="bounds_and_thresholds_nonoverlapping",
        ),
        # Activating a parameter requires a valid inactive region to start with
        param(
            Interval(lower=0.5, upper=1.0),
            Interval(lower=0.5, upper=1.0),
            None,
            "The thresholds must cover zero",
            id="invalid_inactive_region",
        ),
    ],
)
def test_parameter_activation(
    bounds: Interval,
    thresholds: Interval,
    expected_bounds: Interval | None,
    error_message: str,
    mirror: bool,
):
    """The parameter activation utility correctly activates a parameter.

    Args:
        bounds: The bounds of the parameter to activate.
        thresholds: The inactivity thresholds.
        expected_bounds: The expected bounds of the activated parameter.
        error_message: The error message for matching.
        mirror: If ``True``, both bounds and thresholds get mirrored.
    """
    is_valid = expected_bounds is not None

    if mirror:
        bounds = mirror_interval(bounds)
        thresholds = mirror_interval(thresholds)
        if is_valid:
            expected_bounds = mirror_interval(expected_bounds)

    parameter = NumericalContinuousParameter("parameter", bounds=bounds)

    if is_valid:
        activated_parameter = activate_parameter(parameter, thresholds)
        assert activated_parameter.bounds == expected_bounds
        if expected_bounds.is_degenerate:
            assert isinstance(activated_parameter, _FixedNumericalContinuousParameter)
    else:
        with pytest.raises(ValueError, match=error_message):
            activate_parameter(parameter, thresholds)
