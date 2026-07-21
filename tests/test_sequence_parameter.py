"""Tests for SequenceParameter behavior not covered by shared parameter tests."""

from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd
import pytest

from baybe.exceptions import InfiniteParameterError
from baybe.parameters.sequence import SequenceEncoderProtocol, SequenceParameter

_DNA = tuple({"A", "C", "G", "T"})


class _DummyEncoder(SequenceEncoderProtocol):
    def encode(
        self,
        values: nw.Series,
        alphabet: tuple[str, ...],
        *,
        key: str,
        name: str,
    ) -> nw.DataFrame:
        """Passthrough encoder: returns a two-column frame with key and name columns."""
        return values.rename(key).to_frame().with_columns(values.rename(name))


class _MissingJoinKeyEncoder(SequenceEncoderProtocol):
    def encode(
        self,
        values: nw.Series,
        alphabet: tuple[str, ...],
        *,
        key: str,
        name: str,
    ) -> nw.DataFrame:
        """Return only the parameter name column, omitting the join-key column."""
        return values.rename(name).to_frame()


class _MissingNameEncoder(SequenceEncoderProtocol):
    def encode(
        self,
        values: nw.Series,
        alphabet: tuple[str, ...],
        *,
        key: str,
        name: str,
    ) -> nw.DataFrame:
        """Return only the join-key column, omitting the parameter name column."""
        return values.rename(key).to_frame()


class _WrongKeyValuesEncoder(SequenceEncoderProtocol):
    def encode(
        self,
        values: nw.Series,
        alphabet: tuple[str, ...],
        *,
        key: str,
        name: str,
    ) -> nw.DataFrame:
        """Return a join-key column whose values do not match the input."""
        wrong = nw.from_native(pd.Series(["WRONG"], name=key), series_only=True)
        return wrong.to_frame().with_columns(wrong.alias(name))


def test_is_finite_without_max_length():
    """is_finite returns False when no max_length is provided."""
    p = SequenceParameter(name="seq", alphabet=_DNA, encoder=_DummyEncoder())
    assert not p.is_finite


def test_is_finite_with_max_length():
    """is_finite returns True when max_length is provided."""
    p = SequenceParameter(
        name="seq", alphabet=_DNA, encoder=_DummyEncoder(), max_length=2
    )
    assert p.is_finite


def test_values_raises_without_max_length():
    """Accessing values on an infinite SequenceParameter raises an error."""
    p = SequenceParameter(name="seq", alphabet=_DNA, encoder=_DummyEncoder())
    with pytest.raises(InfiniteParameterError):
        _ = p.values


def test_enumerate_values_raises_without_max_length():
    """Calling _enumerate_values on an infinite SequenceParameter raises an error."""
    p = SequenceParameter(name="seq", alphabet=_DNA, encoder=_DummyEncoder())
    with pytest.raises(InfiniteParameterError):
        _ = p._enumerate_values()


def test_values_enumerates_with_max_length():
    """values() enumerates all single-char sequences when min_length=max_length=1."""
    p = SequenceParameter(
        name="seq",
        alphabet=_DNA,
        encoder=_DummyEncoder(),
        min_length=1,
        max_length=1,
    )
    assert set(p.values) == set(_DNA)


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        pytest.param("A", True, id="single_valid_char"),
        pytest.param("ACGT", True, id="multi_valid_chars"),
        pytest.param("AX", False, id="out_of_alphabet_char"),
        pytest.param("", False, id="empty_string"),
        pytest.param(42, False, id="non_string"),  # type: ignore[arg-type]
    ],
)
def test_is_in_range(item, expected):
    """is_in_range applies character-level alphabet membership check."""
    p = SequenceParameter(name="seq", alphabet=_DNA, encoder=_DummyEncoder())
    assert p.is_in_range(item) is expected


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        pytest.param("A", False, id="below_min_length"),
        pytest.param("AC", True, id="at_min_length"),
        pytest.param("ACG", True, id="above_min_length"),
    ],
)
def test_is_in_range_with_min_length(item, expected):
    """is_in_range respects the minimum length constraint."""
    p = SequenceParameter(
        name="seq", alphabet=_DNA, encoder=_DummyEncoder(), min_length=2
    )
    assert p.is_in_range(item) is expected


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        pytest.param("A", True, id="at_max_length"),
        pytest.param("AC", False, id="above_max_length"),
    ],
)
def test_is_in_range_with_max_length(item, expected):
    """is_in_range respects the maximum length constraint."""
    p = SequenceParameter(
        name="seq", alphabet=_DNA, encoder=_DummyEncoder(), max_length=1
    )
    assert p.is_in_range(item) is expected


@pytest.mark.parametrize(
    ("encoder", "match"),
    [
        pytest.param(_MissingJoinKeyEncoder(), "__join_key__", id="missing_join_key"),
        pytest.param(
            _MissingNameEncoder(), "encoded representation", id="missing_name_column"
        ),
        pytest.param(
            _WrongKeyValuesEncoder(),
            "exactly the input values",
            id="wrong_key_values",
        ),
    ],
)
def test_encoding_table_invalid_encoder_contract(encoder, match):
    """Encoders violating the join-key contract raise ValueError during transform."""
    p = SequenceParameter(name="seq", alphabet=_DNA, encoder=encoder)
    with pytest.raises(ValueError, match=match):
        p.transform(["A"])


def test_summary_without_max_length():
    """Summary omits MaxLength and nValues for infinite SequenceParameter."""
    p = SequenceParameter(name="seq", alphabet=_DNA, encoder=_DummyEncoder())
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
        encoder=_DummyEncoder(),
        min_length=1,
        max_length=1,
    )
    s = p.summary()
    assert s["Name"] == "seq"
    assert s["Type"] == "SequenceParameter"
    assert "Alphabet" in s
    assert s["MaxLength"] == 1
    assert s["nValues"] == len(_DNA)
