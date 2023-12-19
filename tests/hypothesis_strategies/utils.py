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

    # Create interval from ordered pair of floats
    bounds = (
        st.tuples(st.floats(), st.floats()).map(sorted).filter(lambda x: x[0] < x[1])
    )
    interval = Interval.create(draw(bounds))

    # Filter excluded intervals
    if exclude_bounded:
        assume(not interval.is_bounded)
    if exclude_half_bounded:
        assume(not interval.is_half_bounded)
    if exclude_fully_unbounded:
        assume(not interval.is_fully_unbounded)

    return interval
