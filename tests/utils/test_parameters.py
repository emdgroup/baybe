"""Tests for parameter utilities."""

from unittest.mock import Mock

import pytest
from pytest import param

from baybe.parameters import NumericalContinuousParameter
from baybe.parameters.utils import activate_parameter
from baybe.utils.interval import Interval


def mirror_interval(interval: Interval) -> Interval:
    """Return an interval copy mirrored around the origin."""
    return Interval(lower=-interval.upper, upper=-interval.lower)


@pytest.mark.parametrize("mirror", [False, True], ids=["regular", "mirrored"])
@pytest.mark.parametrize(
    ("bounds", "thresholds", "expected_result"),
    [
        # Valid case: When parameter lower/upper bounds spread in both
        # negative/positive directions.
        param(
            Interval(lower=-1.0, upper=1.0),
            Interval(lower=-0.5, upper=0.5),
            Interval(lower=-1.0, upper=1.0),
            id="valid_thresholds_in_bounds",
        ),
        # Valid case: When one parameter bound is zero.
        param(
            Interval(lower=-1.0, upper=0.0),
            Interval(lower=-0.5, upper=0.0),
            Interval(lower=-1.0, upper=-0.5),
            id="valid_thresholds_in_bounds_one_side_match_with_zero_threshold",
        ),
        # Invalid case: Activating a parameter requires a valid inactive region to
        # start with.
        param(
            None,
            Interval(lower=0.5, upper=1.0),
            ValueError,
            id="invalid_inactive_region",
        ),
        # Invalid cases related to NotImplementedError. To test whether we can
        # correctly catch different NotImplementedError cases, both nonzero/zero
        # thresholds are considered when necessary.
        param(
            Interval(lower=-1.0, upper=1.0),
            Interval(lower=-0.5, upper=1.0),
            NotImplementedError,
            id="invalid_thresholds_in_bounds_one_side_match_with_nonzero_threshold",
        ),
        param(
            Interval(lower=-1.0, upper=1.0),
            Interval(lower=-1.0, upper=1.0),
            NotImplementedError,
            id="invalid_bounds_on_thresholds_with_nonzero_threshold",
        ),
        param(
            Interval(lower=0.0, upper=1.0),
            Interval(lower=0.0, upper=1.0),
            NotImplementedError,
            id="invalid_bounds_on_thresholds_with_zero_threshold",
        ),
        param(
            Interval(lower=-1.0, upper=1.0),
            Interval(lower=-1.5, upper=1.5),
            NotImplementedError,
            id="invalid_bounds_in_thresholds",
        ),
        param(
            Interval(lower=-1.0, upper=1.0),
            Interval(lower=-1.5, upper=1.0),
            NotImplementedError,
            id="invalid_bounds_in_thresholds_one_side_match_with_nonzero_threshold",
        ),
        param(
            Interval(lower=-1.0, upper=0.0),
            Interval(lower=-1.5, upper=0.0),
            NotImplementedError,
            id="invalid_bounds_in_thresholds_one_side_match_with_zero_threshold",
        ),
        param(
            Interval(lower=-0.5, upper=1.0),
            Interval(lower=-1.0, upper=0.5),
            NotImplementedError,
            id="invalid_bounds_intersected_with_thresholds",
        ),
        param(
            Interval(lower=0.5, upper=1.0),
            Interval(lower=-1.0, upper=0.5),
            NotImplementedError,
            id="invalid_bounds_intersected_with_thresholds_on_nonzero_one_point",
        ),
        param(
            Interval(lower=0.0, upper=1.0),
            Interval(lower=-1.0, upper=0.0),
            NotImplementedError,
            id="invalid_bounds_intersected_with_thresholds_on_zero_one_point",
        ),
        param(
            Interval(lower=0.5, upper=1.0),
            Interval(lower=-1.0, upper=0.0),
            NotImplementedError,
            id="invalid_bounds_and_thresholds_nonoverlapping",
        ),
    ],
)
def test_parameter_activation(
    bounds: Interval | None,
    thresholds: Interval,
    expected_result: Interval | type[ValueError] | type[NotImplementedError],
    mirror: bool,
):
    """The parameter activation utility correctly activates a parameter.

    Args:
        bounds: The bounds of the parameter to activate.
        thresholds: The inactivity thresholds.
        expected_result: The expected bounds of the activated parameter or any error
            raised.
        mirror: If ``True``, both bounds and thresholds get mirrored.
    """
    is_valid = isinstance(expected_result, Interval)

    if mirror:
        thresholds = mirror_interval(thresholds)
        if bounds is not None:
            bounds = mirror_interval(bounds)
        if is_valid:
            expected_result = mirror_interval(expected_result)

    if bounds is None:
        parameter = Mock()
    else:
        parameter = NumericalContinuousParameter("parameter", bounds=bounds)

    if is_valid:
        activated_parameter = activate_parameter(parameter, thresholds)
        assert activated_parameter.bounds == expected_result
    else:
        error_message = (
            "The thresholds must cover zero"
            if expected_result is ValueError
            else "proper sub-interval"
        )
        with pytest.raises(expected_result, match=error_message):
            activate_parameter(parameter, thresholds)
