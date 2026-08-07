"""Tests for SequenceParameter behavior not covered by shared parameter tests."""

from __future__ import annotations

import pytest

from baybe.exceptions import InfiniteSpaceError
from baybe.parameters.sequence import SequenceParameter

_DNA = ("A", "C", "G", "T")
_encoder = lambda series: series.to_frame()  # noqa: E731


@pytest.mark.parametrize(
    ("range", "expected"),
    [
        ((1, 1), 4),
        ((1, 2), 20),
        ((2, 2), 16),
        ((1, 5), 1364),
        ((6, 6), 4096),
    ],
)
def test_length(range, expected):
    """The parameter correctly computes the number of values in its range."""
    p = SequenceParameter(
        name="seq",
        alphabet=_DNA,
        min_length=range[0],
        max_length=range[1],
        encoder=_encoder,
    )
    assert len(p) == expected


@pytest.mark.parametrize("max_length", [None, 1])
def test_is_finite(max_length):
    """The parameter correctly indicates if it is finite or infinite."""
    p = SequenceParameter("seq", _DNA, max_length=max_length, encoder=_encoder)
    assert p.is_finite is (max_length is not None)


def test_values_raises_without_max_length():
    """Accessing finite-based properties of an infinite parameter raises an error."""
    p = SequenceParameter(name="seq", alphabet=_DNA, encoder=_encoder)
    with pytest.raises(InfiniteSpaceError):
        p.values
    with pytest.raises(InfiniteSpaceError):
        len(p)


def test_values_construction():
    """The generated sequences match with the specified range."""
    p = SequenceParameter(
        name="seq",
        alphabet=("A", "BC", "D"),
        encoder=_encoder,
        min_length=1,
        max_length=2,
    )
    assert p.values == (
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


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        pytest.param(("A", "C", "G", "T"), True, id="valid"),
        pytest.param(("A", "C", "G", "T"), True, id="valid_non_tuple"),
        pytest.param("ACGT", False, id="plain_string"),
        pytest.param(42, False, id="wrong_type"),
        pytest.param(("A", "X"), False, id="out_of_alphabet"),
        pytest.param(("A", "C", "G", "T", "A"), False, id="too_long"),
        pytest.param(("A",), False, id="too_short"),
    ],
)
def test_is_in_range(item, expected):
    """In-range check validates element-level alphabet membership and length."""
    p = SequenceParameter("seq", _DNA, encoder=_encoder, min_length=2, max_length=4)
    assert p.is_in_range(item) is expected


def test_equality_modulo_alphabet_ordering():
    """Alphabet ordering does not affect equality of parameters."""
    p1 = SequenceParameter("seq", ("A", "C", "G", "T"), encoder=_encoder)
    p2 = SequenceParameter("seq", ("T", "G", "C", "A"), encoder=_encoder)
    assert p1 == p2
