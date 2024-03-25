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
            "constructor": "from_records",
            "data": [
                {"col_1": 3, "col_2": "a"},
                {"col_1": 2, "col_2": "b"},
                {"col_1": 1, "col_2": "c"},
            ],
        },
        {
            "constructor": "from_records",
            "data": [[1, "a"], [2, "b"], [3, "c"]],
            "columns": ["number_col", "string_col"],
        },
        {
            "constructor": "from_dict",
            "data": {"col_1": [3, 2, 1, 0], "col_2": ["a", "b", "c", "d"]},
        },
        {
            "constructor": "from_dict",
            "data": {"row_1": [0, "a", 1, "b"], "row_2": [2, "b", 3, "d"]},
            "orient": "index",
            "columns": ["col_1", "col_2", "col_3", "col_4"],
        },
        {
            "constructor": "from_dict",
            "data": {
                "index": [("a", "b"), ("a", "c")],
                "columns": [("x", 1), ("y", 2)],
                "data": [[1, 3], [2, 4]],
                "index_names": ["n1", "n2"],
                "column_names": ["z1", "z2"],
            },
            "orient": "tight",
        },
    ],
)
def test_dataframe_constructors(obj):
    deserialize_dataframe(obj)
