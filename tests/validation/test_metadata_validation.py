"""Validation tests for metadata."""

import pytest
from pytest import param

from baybe.utils.metadata import Metadata, to_metadata


@pytest.mark.parametrize(
    ("description", "unit", "misc", "match"),
    [
        param(0, None, None, "must be <class 'str'>", id="desc-non-str"),
        param(None, 0, None, "must be <class 'str'>", id="unit-non-str"),
        param(None, None, 0, "must be <class 'dict'>", id="misc-non-dict"),
        param(None, None, {0: 0}, "must be <class 'str'>", id="misc-non-str-keys"),
    ],
)
def test_invalid_arguments(description, unit, misc, match):
    """Providing invalid arguments raises an error."""
    with pytest.raises(TypeError, match=match):
        Metadata(description, unit, misc)


@pytest.mark.parametrize(
    "invalid_input",
    [
        param("string", id="string_input"),
        param(123, id="number_input"),
        param(["list"], id="list_input"),
        param(object(), id="object_input"),
    ],
)
def test_invalid_input_conversion(invalid_input):
    """Converting invalid inputs raises an error."""
    with pytest.raises(
        TypeError, match="must be a dictionary or a 'Metadata' instance."
    ):
        to_metadata(invalid_input)
