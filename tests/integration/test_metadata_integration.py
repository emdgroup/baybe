"""Integration tests for metadata with BayBE components."""

import pytest

from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.utils.metadata import Metadata


class TestParameterMetadataIntegration:
    """Tests for metadata integration with Parameter class."""

    @pytest.mark.parametrize("as_dict", [True, False])
    def test_parameter_with_metadata(self, as_dict: bool):
        """Parameters accept, ingest, and surface metadata."""
        meta = Metadata(description="test", unit="m", misc={"key": "value"})
        param = NumericalDiscreteParameter(
            name="p",
            values=(1, 2),
            metadata=meta.to_dict() if as_dict else meta,
        )
        assert param.description == "test"
        assert param.unit == "m"
        assert param.metadata.misc == {"key": "value"}

    def test_parameter_without_metadata(self):
        """Parameters without metadata have ``None`` properties."""
        param = NumericalDiscreteParameter(name="p", values=(1, 2))
        assert param.metadata is None
        assert param.description is None
        assert param.unit is None
