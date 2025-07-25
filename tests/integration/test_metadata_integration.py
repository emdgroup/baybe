"""Integration tests for metadata with BayBE components."""

import pytest

from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.targets.enum import TargetMode
from baybe.targets.numerical import NumericalTarget
from baybe.utils.metadata import MeasurableMetadata, Metadata


class TestMeasurableMetadataIntegration:
    """Tests for metadata integration with Parameter class."""

    @pytest.mark.parametrize("as_dict", [True, False])
    def test_parameter_with_metadata(self, as_dict: bool):
        """Parameters accept, ingest, and surface metadata."""
        meta = MeasurableMetadata(description="test", unit="m", misc={"key": "value"})
        param = NumericalDiscreteParameter(
            name="p",
            values=(1, 2),
            metadata=meta.to_dict() if as_dict else meta,
        )
        assert param.description == "test"
        assert param.unit == "m"
        assert param.metadata is not None
        assert param.metadata.misc == {"key": "value"}

    def test_parameter_without_metadata(self):
        """Parameters without metadata have empty metadata and ``None`` properties."""
        param = NumericalDiscreteParameter(name="p", values=(1, 2))
        assert param.metadata is not None
        assert param.metadata.is_empty
        assert param.description is None
        assert param.unit is None


class TestTargetMetadataIntegration:
    """Tests for metadata integration with Target class."""

    def _create_target(self, **kwargs) -> NumericalTarget:
        """Create a standard NumericalTarget for testing."""
        return NumericalTarget(
            name="yield",
            mode=TargetMode.MAX,
            transformation=None,
            **kwargs,
        )

    @pytest.mark.parametrize("as_dict", [True, False])
    def test_target_with_metadata(self, as_dict: bool):
        """Targets accept, ingest, and surface metadata."""
        meta = MeasurableMetadata(
            description="test target", unit="kg", misc={"priority": "high"}
        )
        target = self._create_target(metadata=meta.to_dict() if as_dict else meta)
        assert target.description == "test target"
        assert target.unit == "kg"
        assert target.metadata is not None
        assert target.metadata.misc == {"priority": "high"}

    def test_target_without_metadata(self):
        """Targets without metadata have ``None`` properties."""
        target = self._create_target()
        assert target.metadata is not None
        assert target.metadata.is_empty
        assert target.description is None
        assert target.unit is None


class TestObjectiveMetadataIntegration:
    """Tests for metadata integration with Objective class."""

    def _create_target(self, **kwargs) -> NumericalTarget:
        """Create a standard NumericalTarget for testing."""
        return NumericalTarget(
            name="yield",
            mode=TargetMode.MAX,
            transformation=None,
            **kwargs,
        )

    @pytest.mark.parametrize("as_dict", [True, False])
    def test_objective_with_metadata(self, as_dict: bool):
        """Objectives accept, ingest, and surface metadata."""
        meta = Metadata(
            description="test objective", misc={"algorithm": "GP", "priority": "high"}
        )

        # Create a target first
        target = self._create_target()

        objective = SingleTargetObjective(
            target=target,
            metadata=meta.to_dict() if as_dict else meta,
        )
        assert objective.description == "test objective"
        assert objective.metadata is not None
        assert objective.metadata.misc == {"algorithm": "GP", "priority": "high"}

    def test_objective_without_metadata(self):
        """Objectives without metadata have ``None`` properties."""
        target = self._create_target()

        objective = SingleTargetObjective(target=target)
        assert objective.metadata is not None
        assert objective.metadata.is_empty
        assert objective.description is None

    def test_combined_target_objective_metadata(self):
        """Both target and objective can have independent metadata."""
        target = self._create_target(
            metadata={"description": "Chemical yield", "unit": "%"}
        )

        objective = SingleTargetObjective(
            target=target,
            metadata={"description": "Maximize yield objective"},
        )

        # Both should have their own metadata
        assert target.description == "Chemical yield"
        assert target.unit == "%"
        assert objective.description == "Maximize yield objective"
