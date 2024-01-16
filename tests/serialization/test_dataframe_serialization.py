"""Test serialization of dataframes."""

import hypothesis.strategies as st
import pandas as pd
from hypothesis import given
from hypothesis.extra.pandas import column, data_frames

from baybe.serialization import deserialize_dataframe, serialize_dataframe

_index_strategy = st.one_of(st.text(), st.integers(), st.floats())
"""A strategy for generating dataframe indexes."""


@st.composite
def random_dataframes(draw: st.DrawFn):
    """Generate pandas dataframes of random shape and content."""
    # Generate the dataframe shape
    num_rows = draw(st.integers(min_value=0, max_value=3))
    num_cols = draw(st.integers(min_value=0, max_value=10))

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


@given(random_dataframes())
def test_dataframe_roundtrip(df: pd.DataFrame):
    """A serialization roundtrip yields an equivalent object."""
    string = serialize_dataframe(df)
    df2 = deserialize_dataframe(string)
    assert df.equals(df2), (df, df2)
