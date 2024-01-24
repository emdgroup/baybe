"""Test serialization of dataframes."""

import os

import pandas as pd
from hypothesis import HealthCheck, given, settings

from baybe.serialization import deserialize_dataframe, serialize_dataframe
from tests.hypothesis_strategies.dataframes import random_dataframes


@given(random_dataframes())
@settings(suppress_health_check=[HealthCheck.too_slow] if "CI" in os.environ else [])
def test_dataframe_roundtrip(df: pd.DataFrame):
    """A serialization roundtrip yields an equivalent object."""
    string = serialize_dataframe(df)
    df2 = deserialize_dataframe(string)
    assert df.equals(df2), (df, df2)
