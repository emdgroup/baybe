"""Integration tests for metadata with BayBE components."""

import pytest

from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.targets.enum import TargetMode, TargetTransformation
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

    def test_parameter_metadata_always_non_none(self):
        """Test that parameter metadata is always non-None."""
        # Test parameter without explicit metadata
        param1 = NumericalDiscreteParameter(name="test1", values=(1, 2, 3))
        assert param1.metadata is not None
        assert param1.metadata.is_empty
        assert param1.description is None
        assert param1.unit is None

        # Test parameter with empty dict metadata
        param2 = NumericalDiscreteParameter(name="test2", values=(1, 2, 3), metadata={})
        assert param2.metadata is not None
        assert param2.metadata.is_empty
        assert param2.description is None
        assert param2.unit is None

        # Test parameter with actual metadata
        param3 = NumericalDiscreteParameter(
            name="test3",
            values=(1, 2, 3),
            metadata={"description": "Test param", "unit": "kg"},
        )
        assert param3.metadata is not None
        assert not param3.metadata.is_empty
        assert param3.description == "Test param"
        assert param3.unit == "kg"


class TestTargetMetadataIntegration:
    """Tests for metadata integration with Target class."""

    @pytest.mark.parametrize("as_dict", [True, False])
    def test_target_with_metadata(self, as_dict: bool):
        """Targets accept, ingest, and surface metadata."""
        meta = MeasurableMetadata(
            description="test target", unit="kg", misc={"priority": "high"}
        )
        target = NumericalTarget(
            name="yield",
            bounds=(0, 100),
            mode=TargetMode.MAX,
            transformation=TargetTransformation.LINEAR,
            metadata=meta.to_dict() if as_dict else meta,
        )
        assert target.description == "test target"
        assert target.unit == "kg"
        assert target.metadata is not None
        assert target.metadata.misc == {"priority": "high"}

    def test_target_without_metadata(self):
        """Targets without metadata have ``None`` properties."""
        target = NumericalTarget(
            name="yield",
            bounds=(0, 100),
            mode=TargetMode.MAX,
            transformation=TargetTransformation.LINEAR,
        )
        assert target.metadata is not None
        assert target.metadata.is_empty
        assert target.description is None
        assert target.unit is None

    def test_target_metadata_always_non_none(self):
        """Test that target metadata is always non-None."""
        # Test target without explicit metadata
        target1 = NumericalTarget(
            name="yield",
            bounds=(0, 100),
            mode=TargetMode.MAX,
            transformation=TargetTransformation.LINEAR,
        )
        assert target1.metadata is not None
        assert target1.metadata.is_empty
        assert target1.description is None
        assert target1.unit is None

        # Test target with empty dict metadata
        target2 = NumericalTarget(
            name="yield",
            bounds=(0, 100),
            mode=TargetMode.MAX,
            transformation=TargetTransformation.LINEAR,
            metadata={},
        )
        assert target2.metadata is not None
        assert target2.metadata.is_empty
        assert target2.description is None
        assert target2.unit is None

        # Test target with actual metadata
        target3 = NumericalTarget(
            name="yield",
            bounds=(0, 100),
            mode=TargetMode.MAX,
            transformation=TargetTransformation.LINEAR,
            metadata={"description": "Chemical yield", "unit": "%"},
        )
        assert target3.metadata is not None
        assert not target3.metadata.is_empty
        assert target3.description == "Chemical yield"
        assert target3.unit == "%"

    def test_target_metadata_serialization(self):
        """Target metadata survives serialization roundtrip."""
        target = NumericalTarget(
            name="yield",
            bounds=(0, 100),
            mode=TargetMode.MAX,
            transformation=TargetTransformation.LINEAR,
            metadata={"description": "Chemical yield", "unit": "%", "method": "HPLC"},
        )

        # Serialize and deserialize
        json_str = target.to_json()
        target_restored = NumericalTarget.from_json(json_str)

        # Check metadata is preserved
        assert target_restored.description == "Chemical yield"
        assert target_restored.unit == "%"
        assert target_restored.metadata is not None
        assert target_restored.metadata.misc == {"method": "HPLC"}


class TestObjectiveMetadataIntegration:
    """Tests for metadata integration with Objective class."""

    @pytest.mark.parametrize("as_dict", [True, False])
    def test_objective_with_metadata(self, as_dict: bool):
        """Objectives accept, ingest, and surface metadata."""
        meta = Metadata(
            description="test objective", misc={"algorithm": "GP", "priority": "high"}
        )

        # Create a target first
        target = NumericalTarget(
            name="yield",
            bounds=(0, 100),
            mode=TargetMode.MAX,
            transformation=TargetTransformation.LINEAR,
        )

        objective = SingleTargetObjective(
            target=target,
            metadata=meta.to_dict() if as_dict else meta,
        )
        assert objective.description == "test objective"
        assert objective.metadata is not None
        assert objective.metadata.misc == {"algorithm": "GP", "priority": "high"}

    def test_objective_without_metadata(self):
        """Objectives without metadata have ``None`` properties."""
        target = NumericalTarget(
            name="yield",
            bounds=(0, 100),
            mode=TargetMode.MAX,
            transformation=TargetTransformation.LINEAR,
        )

        objective = SingleTargetObjective(target=target)
        assert objective.metadata is not None
        assert objective.metadata.is_empty
        assert objective.description is None

    def test_objective_metadata_serialization(self):
        """Objective metadata survives serialization roundtrip."""
        target = NumericalTarget(
            name="yield",
            bounds=(0, 100),
            mode=TargetMode.MAX,
            transformation=TargetTransformation.LINEAR,
        )

        objective = SingleTargetObjective(
            target=target,
            metadata={"description": "Maximize yield", "priority": "critical"},
        )

        # Serialize and deserialize
        json_str = objective.to_json()
        objective_restored = SingleTargetObjective.from_json(json_str)

        # Check metadata is preserved
        assert objective_restored.description == "Maximize yield"
        assert objective_restored.metadata is not None
        assert objective_restored.metadata.misc == {"priority": "critical"}

    def test_combined_target_objective_metadata(self):
        """Both target and objective can have independent metadata."""
        target = NumericalTarget(
            name="yield",
            bounds=(0, 100),
            mode=TargetMode.MAX,
            transformation=TargetTransformation.LINEAR,
            metadata={"description": "Chemical yield", "unit": "%"},
        )

        objective = SingleTargetObjective(
            target=target,
            metadata={"description": "Maximize yield objective"},
        )

        # Both should have their own metadata
        assert target.description == "Chemical yield"
        assert target.unit == "%"
        assert objective.description == "Maximize yield objective"

        # Serialize and verify both metadata survive
        json_str = objective.to_json()
        objective_restored = SingleTargetObjective.from_json(json_str)

        assert objective_restored.targets[0].description == "Chemical yield"
        assert objective_restored.targets[0].unit == "%"
        assert objective_restored.description == "Maximize yield objective"

    def test_objective_metadata_always_non_none(self):
        """Test that objective metadata is always non-None."""
        target = NumericalTarget(
            name="yield",
            bounds=(0, 100),
            mode=TargetMode.MAX,
            transformation=TargetTransformation.LINEAR,
        )

        # Test objective without explicit metadata
        objective1 = SingleTargetObjective(target=target)
        assert objective1.metadata is not None
        assert objective1.metadata.is_empty
        assert objective1.description is None

        # Test objective with empty dict metadata
        objective2 = SingleTargetObjective(target=target, metadata={})
        assert objective2.metadata is not None
        assert objective2.metadata.is_empty
        assert objective2.description is None

        # Test objective with actual metadata
        objective3 = SingleTargetObjective(
            target=target,
            metadata={"description": "Maximize yield", "priority": "critical"},
        )
        assert objective3.metadata is not None
        assert not objective3.metadata.is_empty
        assert objective3.description == "Maximize yield"
