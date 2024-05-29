"""Hypothesis strategies for dataframes."""

from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, indexes

from ..hypothesis_strategies.basic import finite_floats


@st.composite
def random_dataframes(draw: st.DrawFn):
    """Generate pandas dataframes of random shape and content."""
    index_elements = st.one_of(st.text(), st.integers(), finite_floats())
    cols = st.builds(
        column, name=index_elements, dtype=st.sampled_from([int, float, str])
    )
    columns = draw(st.lists(cols, max_size=5, unique_by=lambda c: c.name))
    index_strategy = indexes(elements=index_elements, max_size=2)
    return draw(data_frames(columns=columns, index=index_strategy))
