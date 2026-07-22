"""Tests for SequenceParameter behavior not covered by shared parameter tests."""

from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest

from baybe.exceptions import InfiniteSpaceError
from baybe.parameters.base import _JOIN_KEY
from baybe.parameters.sequence import SequenceParameter

_DNA = ("A", "C", "G", "T")


def _dummy_encoder(series: nw.Series) -> nw.DataFrame:
    """Passthrough encoder: returns the input series as a single-column frame."""
    return series.to_frame()


def test_is_finite_without_max_length():
    """is_finite returns False when no max_length is provided."""
    p = SequenceParameter(name="seq", alphabet=_DNA, encoder=_dummy_encoder)
    assert not p.is_finite


def test_is_finite_with_max_length():
    """is_finite returns True when max_length is provided."""
    p = SequenceParameter(
        name="seq", alphabet=_DNA, encoder=_dummy_encoder, max_length=2
    )
    assert p.is_finite


def test_values_construction():
    """Test sequence values construction."""
    p = SequenceParameter(
        name="seq",
        alphabet=("A", "BC", "D"),
        encoder=_dummy_encoder,
        max_length=2,
        min_length=1,
    )
    all_values = p._enumerate_values
    assert all_values == (
        ("A",),
        ("BC",),
        ("D",),
        ("A", "A"),
        ("A", "BC"),
        ("A", "D"),
        ("BC", "A"),
        ("BC", "BC"),
        ("BC", "D"),
        ("D", "A"),
        ("D", "BC"),
        ("D", "D"),
    )


def test_values_raises_without_max_length():
    """Accessing values on an infinite SequenceParameter raises an error."""
    p = SequenceParameter(name="seq", alphabet=_DNA, encoder=_dummy_encoder)
    with pytest.raises(InfiniteSpaceError):
        _ = p.values


def test_enumerate_values_raises_without_max_length():
    """Accessing _enumerate_values on an infinite SequenceParameter raises an error."""
    p = SequenceParameter(name="seq", alphabet=_DNA, encoder=_dummy_encoder)
    with pytest.raises(InfiniteSpaceError):
        _ = p._enumerate_values


def test_values_enumerates_with_max_length():
    """Values enumerates all single-element sequences when min_length=max_length=1."""
    p = SequenceParameter(
        name="seq",
        alphabet=_DNA,
        encoder=_dummy_encoder,
        min_length=1,
        max_length=1,
    )
    assert set(p.values) == {(ch,) for ch in _DNA}


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        pytest.param(("A",), True, id="single_valid_element"),
        pytest.param(("A", "C", "G", "T"), True, id="multi_valid_elements"),
        pytest.param(("A", "X"), False, id="out_of_alphabet_element"),
        pytest.param(42, False, id="non_tuple"),  # type: ignore[arg-type]
    ],
)
def test_is_in_range(item, expected):
    """is_in_range applies element-level alphabet membership check."""
    p = SequenceParameter(name="seq", alphabet=_DNA, encoder=_dummy_encoder)
    assert p.is_in_range(item) is expected


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        pytest.param(("A",), False, id="below_min_length"),
        pytest.param(("A", "C"), True, id="at_min_length"),
        pytest.param(("A", "C", "G"), True, id="above_min_length"),
    ],
)
def test_is_in_range_with_min_length(item, expected):
    """is_in_range respects the minimum length constraint."""
    p = SequenceParameter(
        name="seq", alphabet=_DNA, encoder=_dummy_encoder, min_length=2
    )
    assert p.is_in_range(item) is expected


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        pytest.param(("A",), True, id="at_max_length"),
        pytest.param(("A", "C"), False, id="above_max_length"),
    ],
)
def test_is_in_range_with_max_length(item, expected):
    """is_in_range respects the maximum length constraint."""
    p = SequenceParameter(
        name="seq", alphabet=_DNA, encoder=_dummy_encoder, max_length=1
    )
    assert p.is_in_range(item) is expected


def test_summary_without_max_length():
    """Summary omits MaxLength and nValues for infinite SequenceParameter."""
    p = SequenceParameter(name="seq", alphabet=_DNA, encoder=_dummy_encoder)
    s = p.summary()
    assert s["Name"] == "seq"
    assert s["Type"] == "SequenceParameter"
    assert "Alphabet" in s
    assert "MaxLength" not in s
    assert "nValues" not in s


def test_summary_with_max_length():
    """Summary includes MaxLength and nValues for finite SequenceParameter."""
    p = SequenceParameter(
        name="seq",
        alphabet=_DNA,
        encoder=_dummy_encoder,
        min_length=1,
        max_length=1,
    )
    s = p.summary()
    assert s["Name"] == "seq"
    assert s["Type"] == "SequenceParameter"
    assert "Alphabet" in s
    assert s["MaxLength"] == 1
    assert s["nValues"] == len(_DNA)


@pytest.mark.parametrize(
    ("series", "dataframe"),
    [
        pytest.param(
            pd.Series(["A", "C"], name="seq"),
            pd.DataFrame({"seq": ["A", "C"]}),
            id="pandas",
        ),
        pytest.param(
            pl.Series("seq", ["A", "C"]),
            pl.DataFrame({"seq": ["A", "C"]}),
            id="polars",
        ),
    ],
)
def test_different_dataframe_backends(series, dataframe):
    """Test that the encoding table works with different DataFrame backends."""
    p = SequenceParameter(
        name="seq",
        alphabet=_DNA,
        encoder=lambda _: dataframe,
        min_length=1,
        max_length=1,
    )
    encoding_table = p._encoding_table(nw.from_native(series, series_only=True))
    assert encoding_table.shape[0] == 2
    assert "seq" in encoding_table.columns
    assert set(encoding_table["seq"]) == {"A", "C"}
    assert _JOIN_KEY in encoding_table.columns
