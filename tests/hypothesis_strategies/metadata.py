"""Hypothesis strategies for metadata."""

import hypothesis.strategies as st
from hypothesis import assume

from baybe.serialization.core import _TYPE_FIELD
from baybe.utils.metadata import MeasurableMetadata, Metadata

_descriptions = st.one_of(st.none(), st.text(min_size=0))
"""A strategy generating metadata descriptions."""


@st.composite
def _miscs(draw: st.DrawFn, cls: type[Metadata]):
    """Generates miscellaneous metadata for various metadata classes."""
    misc = draw(
        st.dictionaries(
            st.text(min_size=0),
            st.one_of(st.text(), st.integers(), st.floats(allow_nan=False)),
            max_size=5,
        )
    )
    assume(not set(misc).intersection(cls._explicit_fields | {_TYPE_FIELD}))
    return misc


@st.composite
def metadata(draw: st.DrawFn):
    """Generate :class:`baybe.utils.metadata.Metadata`."""
    description = draw(_descriptions)
    misc = draw(_miscs(Metadata))
    return Metadata(description=description, misc=misc)


@st.composite
def measurable_metadata(draw: st.DrawFn):
    """Generate :class:`baybe.parameters.base.MeasurableMetadata`."""
    description = draw(_descriptions)
    unit = draw(st.one_of(st.none(), st.text(min_size=0)))
    misc = draw(_miscs(MeasurableMetadata))
    return MeasurableMetadata(description=description, unit=unit, misc=misc)
