"""Hypothesis strategies for dataframes."""

from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames

_index_strategy = st.one_of(st.text(), st.integers(), st.floats())
"""A strategy for generating dataframe indexes."""


@st.composite
def random_dataframes(draw: st.DrawFn):
    """Generate pandas dataframes of random shape and content."""
    # Generate the dataframe shape
    num_rows = draw(st.integers(min_value=0, max_value=2))
    num_cols = draw(st.integers(min_value=0, max_value=5))

    # Generate the column names/types and index
    col_names = draw(
        st.lists(_index_strategy, min_size=num_cols, max_size=num_cols, unique=True)
    )
    col_types = draw(
        st.lists(
            st.sampled_from([int, float, str]), min_size=num_cols, max_size=num_cols
        )
    )
    index = st.lists(_index_strategy, min_size=num_rows, max_size=num_rows)

    # Define the column content
    columns = [
        column(name=name, dtype=dtype) for name, dtype in zip(col_names, col_types)
    ]

    # Generate the dataframe
    return draw(data_frames(columns=columns, index=index))
