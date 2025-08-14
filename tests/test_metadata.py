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

    @pytest.mark.parametrize(
        ("metadata", "expected"),
        [
            param({}, True, id="empty"),
            param({"description": "test"}, False, id="description_only"),
            param({"unit": "kg"}, False, id="unit_only"),
            param({"misc": {"key": "value"}}, False, id="misc_only"),
        ],
    )
    @pytest.mark.parametrize("metadata_cls", [Metadata, MeasurableMetadata])
    def test_metadata_is_empty_detection(self, metadata_cls, metadata, expected):
        """The is_empty property correctly identifies empty metadata."""
        if metadata_cls is Metadata and "unit" in metadata:
            pytest.skip("Metadata class has no 'unit' attribute.")
        assert metadata_cls(**metadata).is_empty == expected


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
