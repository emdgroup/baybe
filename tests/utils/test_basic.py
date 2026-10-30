"""Tests for basic utilities."""

from itertools import islice

import pytest
from pytest import param

from baybe.utils.basic import iter_indices, register_hooks


def f_plain(arg1, arg2):
    pass


def f_annotated(arg1: str, arg2: int):
    pass


def f_annotated_one_default(arg1: str, arg2: int = 1):
    pass


def f2_plain(arg, arg3):
    pass


@pytest.mark.parametrize(
    ("target", "hook"),
    [
        param(
            f_annotated,
            f_annotated_one_default,
            id="hook_with_defaults",
        ),
        param(
            f_annotated_one_default,
            f_annotated,
            id="target_with_defaults",
        ),
        param(
            f_annotated,
            f_plain,
            id="hook_without_annotations",
        ),
    ],
)
def test_valid_register_hooks(target, hook):
    """Passing consistent signatures to `register_hooks` raises no error."""
    register_hooks(target, [hook])


@pytest.mark.parametrize(
    ("target", "hook"),
    [
        param(
            f_annotated,
            f2_plain,
            id="different_names",
        ),
    ],
)
def test_invalid_register_hooks(target, hook):
    """Passing inconsistent signatures to `register_hooks` raises an error."""
    with pytest.raises(TypeError):
        register_hooks(target, [hook])


@pytest.mark.parametrize("n", [0, 5], ids=["n0", "n5"])
def test_iter_indices_ordered(n):
    """Ordered mode yields all indices in ascending order."""
    assert list(iter_indices(n, mode="ordered")) == list(range(n))


@pytest.mark.parametrize("n", [0, 5], ids=["n0", "n5"])
def test_iter_indices_shuffled(n):
    """Shuffled mode yields each index exactly once in arbitrary order."""
    assert sorted(iter_indices(n, mode="shuffled")) == list(range(n))


@pytest.mark.parametrize("n", [1, 5], ids=["n1", "n5"])
def test_iter_indices_with_replacement(n):
    """With-replacement mode yields an infinite stream of valid indices."""
    indices = list(islice(iter_indices(n, mode="with_replacement"), 10))
    assert len(indices) == 10
    assert all(0 <= idx < n for idx in indices)


@pytest.mark.parametrize(
    "mode",
    ["ordered", "shuffled", "with_replacement"],
    ids=["ordered", "shuffled", "with_replacement"],
)
def test_iter_indices_negative_n(mode):
    """Negative index counts are rejected."""
    with pytest.raises(ValueError, match="must be non-negative"):
        list(iter_indices(-1, mode=mode))


def test_iter_indices_with_replacement_empty():
    """With-replacement sampling from an empty collection is rejected."""
    with pytest.raises(ValueError, match="empty collection"):
        list(iter_indices(0, mode="with_replacement"))
