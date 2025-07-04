"""Validation tests for metadata."""

import pytest
from pytest import param

from baybe.utils.metadata import to_metadata


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
