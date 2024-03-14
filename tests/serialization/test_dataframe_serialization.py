"""Test serialization of dataframes."""

import pandas as pd
from hypothesis import given
from pytest import mark

from baybe.serialization import deserialize_dataframe, serialize_dataframe
from baybe.serialization.core import _unstructure_dataframe_hook
from tests.hypothesis_strategies.dataframes import random_dataframes


@given(random_dataframes())
def test_dataframe_roundtrip(df: pd.DataFrame):
    """A serialization roundtrip yields an equivalent object."""
    string = serialize_dataframe(df)
    df2 = deserialize_dataframe(string)
    assert df.equals(df2), (df, df2)


@mark.parametrize(
    "obj",
    [
        _unstructure_dataframe_hook(pd.DataFrame({"c1": [1, 2], "c2": [3, 4]})),
        {
            "constructor": "read_json",
            "path_or_buf": '[{"c1":"a","c2":"b"},{"c1":"c","c2":"d"}]',
            "orient": "records",
        },
        {
            "constructor": "read_json",
            "path_or_buf": '{"r1":{"c1":"a","c2":"b"},"r2":{"c1":"c","c2":"d"}}',
            "orient": "index",
        },
        {
            "constructor": "read_json",
            "path_or_buf": '{"columns":["c1","c2"],"index":["r1","r2"],'
            '"data":[["a","b"],["c","d"]]}',
            "orient": "split",
        },
    ],
)
def test_dataframe_constructors(obj):
    deserialize_dataframe(obj)
