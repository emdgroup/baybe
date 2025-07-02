"""Integration tests for metadata with BayBE components."""

from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.utils.metadata import Metadata


class TestParameterMetadataIntegration:
    """Tests for metadata integration with Parameter class."""

    def test_parameter_with_metadata_dict(self):
        """Test parameter accepts metadata as dict."""
        param = NumericalDiscreteParameter(
            name="test_param",
            values=(1.0, 2.0, 3.0),
            metadata={"description": "test parameter", "unit": "kg"},
        )
        assert param.description == "test parameter"
        assert param.unit == "kg"
        assert param.metadata.misc == {}

    def test_parameter_with_metadata_instance(self):
        """Test parameter accepts Metadata instance."""
        meta = Metadata(description="direct metadata", unit="m")
        param = NumericalDiscreteParameter(
            name="test_param", values=(1.0, 2.0, 3.0), metadata=meta
        )
        assert param.description == "direct metadata"
        assert param.unit == "m"
        assert param.metadata is meta

    def test_parameter_without_metadata(self):
        """Test parameter without metadata has None properties."""
        param = NumericalDiscreteParameter(name="test_param", values=(1.0, 2.0, 3.0))
        assert param.metadata is None
        assert param.description is None
        assert param.unit is None

    def test_parameter_metadata_with_misc(self):
        """Test parameter metadata preserves misc fields."""
        param = NumericalDiscreteParameter(
            name="test_param",
            values=(1.0, 2.0, 3.0),
            metadata={
                "description": "test",
                "unit": "kg",
                "custom_field": "custom_value",
                "priority": 1,
            },
        )
        assert param.description == "test"
        assert param.unit == "kg"
        assert param.metadata.misc == {"custom_field": "custom_value", "priority": 1}
