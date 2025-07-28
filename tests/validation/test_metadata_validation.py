"""Validation tests for metadata."""

import pytest
from pytest import param

from baybe.utils.metadata import MeasurableMetadata, Metadata, to_metadata


@pytest.mark.parametrize(
    ("description", "misc", "error", "match"),
    [
        param(0, {}, TypeError, "must be <class 'str'>", id="desc-non-str"),
        param(None, 0, TypeError, "must be <class 'dict'>", id="misc-non-dict"),
        param(
            None,
            {0: 0},
            TypeError,
            "must be <class 'str'>",
            id="misc-non-str-keys",
        ),
        param(
            None,
            {"description": 0},
            ValueError,
            "fields: {'description'}",
            id="desc_in_misc",
        ),
    ],
)
def test_invalid_arguments_for_metadata(description, misc, error, match):
    """Providing invalid arguments to base Metadata class raises an error."""
    with pytest.raises(error, match=match):
        Metadata(description, misc=misc)


@pytest.mark.parametrize(
    ("description", "unit", "misc", "error", "match"),
    [
        param(None, 0, {}, TypeError, "must be <class 'str'>", id="unit-non-str"),
        param(
            None, None, {"unit": 0}, ValueError, "fields: {'unit'}", id="unit_in_misc"
        ),
    ],
)
def test_invalid_arguments_for_measurable_metadata(
    description, unit, misc, error, match
):
    """Providing invalid arguments raises an error."""
    with pytest.raises(error, match=match):
        MeasurableMetadata(description, unit, misc=misc)


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
        to_metadata(invalid_input, Metadata)
