"""Tests for generic metadata functionality."""

import pytest
from pytest import param

from baybe.utils.metadata import Metadata, to_metadata


class TestMetadata:
    """Tests for Metadata dataclass."""

    def test_metadata_creation_basic(self):
        """Test basic metadata creation with all fields."""
        meta = Metadata(description="test param", unit="kg")
        assert meta.description == "test param"
        assert meta.unit == "kg"
        assert meta.misc == {}

    def test_metadata_creation_minimal(self):
        """Test metadata creation with minimal fields."""
        meta = Metadata()
        assert meta.description is None
        assert meta.unit is None
        assert meta.misc == {}

    def test_metadata_creation_with_misc(self):
        """Test metadata creation with misc data."""
        misc_data = {"custom_field": "value", "number": 42}
        meta = Metadata(description="test", misc=misc_data)
        assert meta.description == "test"
        assert meta.misc == misc_data


class TestMetadataConverter:
    """Tests for to_metadata function."""

    def test_convert_metadata_instance(self):
        """Test converting Metadata instance returns same instance."""
        meta = Metadata(description="test")
        result = to_metadata(meta)
        assert result is meta

    @pytest.mark.parametrize(
        ("input_dict", "expected_desc", "expected_unit", "expected_misc"),
        [
            param(
                {"description": "test desc", "unit": "kg"},
                "test desc",
                "kg",
                {},
                id="known_fields_only",
            ),
            param(
                {"description": "test", "custom_field": "value"},
                "test",
                None,
                {"custom_field": "value"},
                id="description_and_misc",
            ),
            param(
                {"unit": "m", "other": 42}, None, "m", {"other": 42}, id="unit_and_misc"
            ),
            param(
                {"custom1": "val1", "custom2": 123},
                None,
                None,
                {"custom1": "val1", "custom2": 123},
                id="misc_only",
            ),
            param({}, None, None, {}, id="empty_dict"),
        ],
    )
    def test_convert_dict_valid(
        self, input_dict, expected_desc, expected_unit, expected_misc
    ):
        """Test converting valid dict inputs."""
        result = to_metadata(input_dict)
        assert isinstance(result, Metadata)
        assert result.description == expected_desc
        assert result.unit == expected_unit
        assert result.misc == expected_misc

    @pytest.mark.parametrize(
        ("invalid_input", "expected_error"),
        [
            param("string", TypeError, id="string_input"),
            param(123, TypeError, id="number_input"),
            param(["list"], TypeError, id="list_input"),
            param(object(), TypeError, id="object_input"),
        ],
    )
    def test_convert_invalid_input(self, invalid_input, expected_error):
        """Test converting invalid inputs raises TypeError."""
        with pytest.raises(
            expected_error, match="Metadata must be dict or Metadata instance."
        ):
            to_metadata(invalid_input)
