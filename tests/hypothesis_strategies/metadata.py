"""Hypothesis strategies for metadata."""

import hypothesis.strategies as st
from hypothesis import assume

from baybe.utils.metadata import Metadata


@st.composite
def metadata(draw: st.DrawFn):
    """Generate :class:`baybe.utils.metadata.Metadata`."""
    description = draw(st.one_of(st.none(), st.text(min_size=1)))
    unit = draw(st.one_of(st.none(), st.text(min_size=1)))
    misc = draw(
        st.dictionaries(
            st.text(min_size=1),
            st.one_of(st.text(), st.integers(), st.floats(allow_nan=False)),
            max_size=5,
        )
    )
    assume(not Metadata._explicit_fields.intersection(misc))
    return Metadata(description=description, unit=unit, misc=misc)
