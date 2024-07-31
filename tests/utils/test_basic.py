"""Tests for basic utilities."""

import pytest
from pytest import param

from baybe.utils.basic import register_hooks


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
