"""Tests for metadata functionality."""

import pytest
from pytest import param

from baybe.utils.metadata import MeasurableMetadata, Metadata, to_metadata


class TestMetadata:
    """Tests for `Metadata` class."""

    def test_metadata_creation_basic(self):
        """All metadata attributes are properly populated."""
        meta = MeasurableMetadata(
            description="test", unit="kg", misc={"custom_field": "value"}
        )
        assert meta.description == "test"
        assert meta.unit == "kg"
        assert meta.misc == {"custom_field": "value"}

    def test_metadata_creation_defaults(self):
        """Metadata can be created with no content."""
        meta = MeasurableMetadata()
        assert meta.description is None
        assert meta.unit is None
        assert meta.misc == {}

    def test_metadata_is_empty_detection(self):
        """The is_empty property correctly identifies empty metadata."""
        # Test empty MeasurableMetadata
        meta1 = MeasurableMetadata()
        assert meta1.is_empty

        # Test MeasurableMetadata with description only
        meta2 = MeasurableMetadata(description="test")
        assert not meta2.is_empty

        # Test MeasurableMetadata with unit only
        meta3 = MeasurableMetadata(unit="kg")
        assert not meta3.is_empty

        # Test MeasurableMetadata with misc only
        meta4 = MeasurableMetadata(misc={"key": "value"})
        assert not meta4.is_empty

        # Test empty base Metadata
        meta5 = Metadata()
        assert meta5.is_empty

        # Test Metadata with description
        meta6 = Metadata(description="test")
        assert not meta6.is_empty

        # Test Metadata with misc
        meta7 = Metadata(misc={"key": "value"})
        assert not meta7.is_empty


class TestMetadataConverter:
    """Tests for `to_metadata` function."""

    def test_convert_metadata_instance(self):
        """The converter passes through Metadata instances unchanged."""
        meta = MeasurableMetadata(description="test")
        result = to_metadata(meta, MeasurableMetadata)
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
                {"unit": "m", "other": 42},
                None,
                "m",
                {"other": 42},
                id="unit_and_misc",
            ),
            param(
                {"custom1": "val1", "custom2": 123},
                None,
                None,
                {"custom1": "val1", "custom2": 123},
                id="misc_only",
            ),
            param(
                {},
                None,
                None,
                {},
                id="empty_dict",
            ),
        ],
    )
    def test_convert_dict_valid(
        self, input_dict, expected_desc, expected_unit, expected_misc
    ):
        """Conversion from dict properly separates known and unknown attributes."""
        result = to_metadata(input_dict, MeasurableMetadata)
        assert isinstance(result, MeasurableMetadata)
        assert result.description == expected_desc
        assert result.unit == expected_unit
        assert result.misc == expected_misc
