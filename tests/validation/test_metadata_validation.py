"""Validation tests for metadata."""

import pytest
from pytest import param

from baybe.objectives.single import SingleTargetObjective
from baybe.targets.enum import TargetMode
from baybe.targets.numerical import NumericalTarget
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


class TestTargetMetadataValidation:
    """Validation tests for target metadata."""

    def _create_target(self, **kwargs) -> NumericalTarget:
        """Create a standard NumericalTarget for testing."""
        return NumericalTarget(
            name="test",
            mode=TargetMode.MAX,
            transformation=None,
            **kwargs,
        )

    def test_target_invalid_metadata_type(self):
        """Creating target with invalid metadata type raises error."""
        with pytest.raises(
            TypeError, match="must be a dictionary or a 'MeasurableMetadata' instance"
        ):
            self._create_target(metadata="invalid_string")

    def test_target_metadata_with_invalid_unit(self):
        """Target metadata with invalid unit type raises error."""
        with pytest.raises(TypeError, match="must be <class 'str'>"):
            self._create_target(metadata={"unit": 123})  # Invalid unit type

    def test_target_metadata_with_unit_field_separation(self):
        """Target metadata should automatically separate unit from misc."""
        # This test shows that unit is automatically separated from misc
        target = self._create_target(
            metadata={"description": "test", "unit": "kg", "extra": "value"}
        )
        # The unit should be extracted to the unit field
        assert target.unit == "kg"
        assert target.description == "test"
        # Only extra should remain in misc
        assert target.metadata is not None
        assert target.metadata.misc == {"extra": "value"}

    def test_target_direct_metadata_with_unit_in_misc(self):
        """Direct metadata creation with unit in misc should raise error."""
        with pytest.raises(ValueError, match="fields: {'unit'}"):
            MeasurableMetadata(description="test", misc={"unit": "kg"})


class TestObjectiveMetadataValidation:
    """Validation tests for objective metadata."""

    def _create_target(self, **kwargs) -> NumericalTarget:
        """Create a standard NumericalTarget for testing."""
        return NumericalTarget(
            name="test",
            mode=TargetMode.MAX,
            transformation=None,
            **kwargs,
        )

    def test_objective_invalid_metadata_type(self):
        """Creating objective with invalid metadata type raises error."""
        target = self._create_target()

        with pytest.raises(
            TypeError, match="must be a dictionary or a 'Metadata' instance"
        ):
            SingleTargetObjective(target=target, metadata=["invalid_list"])

    def test_objective_metadata_field_separation(self):
        """Objective metadata should automatically separate description from misc."""
        target = self._create_target()

        # Test that description is automatically separated from other fields
        objective = SingleTargetObjective(
            target=target,
            metadata={
                "description": "test objective",
                "priority": "high",
                "algorithm": "GP",
            },
        )

        assert objective.description == "test objective"
        assert objective.metadata is not None
        assert objective.metadata.misc == {"priority": "high", "algorithm": "GP"}

    def test_objective_direct_metadata_with_description_in_misc(self):
        """Direct metadata creation with description in misc should raise error."""
        with pytest.raises(ValueError, match="fields: {'description'}"):
            Metadata(misc={"description": "should not be here"})

    def test_none_metadata_raises_error(self):
        """Test that passing None for metadata raises TypeError."""
        from baybe.parameters.numerical import NumericalDiscreteParameter

        with pytest.raises(
            TypeError, match="must be a dictionary or a 'MeasurableMetadata' instance"
        ):
            NumericalDiscreteParameter(name="test", values=(1, 2, 3), metadata=None)
