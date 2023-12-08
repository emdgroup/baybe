"""Hypothesis strategies for generating utility objects."""

import hypothesis.strategies as st
from hypothesis import assume

from baybe.utils.interval import Interval


@st.composite
def interval(
    draw: st.DrawFn,
    *,
    exclude_bounded: bool = False,
    exclude_half_bounded: bool = False,
    exclude_fully_unbounded: bool = False,
):
    """Generate :class:`baybe.utils.interval.Interval`."""
    assert not all(
        (exclude_bounded, exclude_half_bounded, exclude_fully_unbounded)
    ), "At least one Interval type must be allowed."

    # Create interval from ordered pair of floats / None
    lower = draw(st.none() | st.floats(max_value=float("inf"), exclude_max=True))
    min_value = lower if lower is not None else float("-inf")
    upper = draw(st.none() | st.floats(min_value=min_value, exclude_min=True))
    bounds = [lower, upper]
    if None not in bounds:
        bounds.sort()
    interval = Interval(*bounds)

    # Filter excluded intervals
    if exclude_bounded:
        assume(not interval.is_bounded)
    if exclude_half_bounded:
        assume(not interval.is_half_bounded)
    if exclude_fully_unbounded:
        assume(not interval.is_left_bounded and not interval.is_right_bounded)

    return interval
