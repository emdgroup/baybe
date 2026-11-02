"""Dataframe serialization tests."""

from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from narwhals.testing import assert_frame_equal
from pytest import mark, param

from baybe.serialization import deserialize_dataframe, serialize_dataframe
from baybe.settings import Settings
from tests.hypothesis_strategies.dataframes import random_dataframes

# ---------------------------------------------------------------------------
# Native dataframe roundtrip
# ---------------------------------------------------------------------------


@mark.parametrize(
    "impl",
    [nw.Implementation.PANDAS, nw.Implementation.POLARS],
    ids=["pandas", "polars"],
)
@given(data=st.data())
def test_native_dataframe_roundtrip(impl, data):
    """Serializing and deserializing a native dataframe preserves its data."""
    df = data.draw(random_dataframes(string_column_names_only=True, backend=impl))
    nw_df = nw.from_native(df, eager_only=True)
    assume(len(nw_df.columns) > 0)
    assume(all(col != "" for col in nw_df.columns))

    with Settings(default_dataframe_backend=impl):
        result = deserialize_dataframe(serialize_dataframe(df))

    assert_frame_equal(nw_df, nw.from_native(result, eager_only=True))


# ---------------------------------------------------------------------------
# Constructor-based deserialization
# ---------------------------------------------------------------------------


@mark.parametrize(
    "impl",
    [nw.Implementation.PANDAS, nw.Implementation.POLARS],
    ids=["pandas", "polars"],
)
@mark.parametrize(
    "spec",
    [
        param(
            {"constructor": "from_dict", "data": {"x": [1, 3], "y": [2, 4]}},
            id="from_dict",
        ),
        param(
            {
                "constructor": "from_dicts",
                "data": [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
            },
            id="from_dicts",
        ),
    ],
)
def test_constructor_based_deserialization(spec, impl):
    """Dict-based deserialization dispatches to the correct narwhals constructor.

    The result's data matches the input and its native backend matches the active
    dataframe backend.
    """
    with Settings(default_dataframe_backend=impl):
        result = deserialize_dataframe(dict(spec))

    result_nw = nw.from_native(result, eager_only=True)
    assert result_nw.implementation == impl
    assert_frame_equal(
        result_nw,
        nw.from_dict({"x": [1, 3], "y": [2, 4]}, backend=impl),
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@mark.parametrize(
    "df",
    [
        param(pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30]), id="integer-index"),
        param(pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"]), id="string-index"),
        param(pd.DataFrame({"a": [1, 2, 3]}).iloc[1:], id="non-default-range-index"),
    ],
)
def test_serialize_non_trivial_index_raises(df):
    """Serializing a pandas DataFrame with a non-trivial index raises."""
    with pytest.raises(ValueError, match="non-default index"):
        serialize_dataframe(df)


def test_missing_constructor_key_raises():
    """A dict without a ``constructor`` key raises a descriptive error."""
    with pytest.raises(ValueError, match="constructor"):
        deserialize_dataframe({"data": {"col_1": [1, 2]}})


def test_unknown_constructor_raises():
    """An unknown constructor name raises an AttributeError."""
    with pytest.raises(AttributeError):
        deserialize_dataframe({"constructor": "nonexistent", "data": {}})
