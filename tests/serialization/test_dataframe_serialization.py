"""Test serialization of dataframes."""

import pandas as pd
from hypothesis import given

from baybe.serialization import deserialize_dataframe, serialize_dataframe
from tests.hypothesis_strategies.dataframes import random_dataframes


@given(random_dataframes())
def test_dataframe_roundtrip(df: pd.DataFrame):
    """A serialization roundtrip yields an equivalent object."""
    string = serialize_dataframe(df)
    df2 = deserialize_dataframe(string)
    assert df.equals(df2), (df, df2)
