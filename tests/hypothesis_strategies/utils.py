"""Hypothesis strategies for generating utility objects."""

from enum import Enum, auto

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st

from baybe.utils.interval import Interval


class IntervalType(Enum):
    """The possible types of an interval on the real number line."""

    FULLY_UNBOUNDED = auto()
    HALF_BOUNDED = auto()
    BOUNDED = auto()


@st.composite
def intervals(
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

    # Draw the interval type from the allowed types
    type_gate = {
        IntervalType.FULLY_UNBOUNDED: not exclude_fully_unbounded,
        IntervalType.HALF_BOUNDED: not exclude_half_bounded,
        IntervalType.BOUNDED: not exclude_bounded,
    }
    allowed_types = [t for t, b in type_gate.items() if b]
    interval_type = draw(st.sampled_from(allowed_types))

    # A strategy producing finite floats
    ffloats = st.floats(allow_infinity=False, allow_nan=False)

    # Draw the bounds depending on the interval type
    if interval_type is IntervalType.FULLY_UNBOUNDED:
        bounds = (None, None)
    elif interval_type is IntervalType.HALF_BOUNDED:
        bounds = draw(
            st.sampled_from(
                [
                    (None, draw(ffloats)),
                    (draw(ffloats), None),
                ]
            )
        )
    elif interval_type is IntervalType.BOUNDED:
        bounds = draw(
            hnp.arrays(
                dtype=float,
                shape=(2,),
                elements=ffloats,
                unique=True,
            ).map(sorted)
        )
    else:
        raise RuntimeError("This line should be unreachable.")

    return Interval.create(bounds)
