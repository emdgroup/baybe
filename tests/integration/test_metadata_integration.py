"""Integration tests for metadata with BayBE components."""

import pytest
from pytest import param

from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.targets.enum import TargetMode
from baybe.targets.numerical import NumericalTarget
from baybe.utils.metadata import MeasurableMetadata, Metadata, to_metadata

TMetadata = Metadata | dict | None


def make_parameter(metadata: TMetadata = None) -> NumericalDiscreteParameter:
    return NumericalDiscreteParameter(
        "p", (1, 2), metadata=metadata or MeasurableMetadata()
    )


def make_target(metadata: TMetadata = None) -> NumericalTarget:
    return NumericalTarget(
        "yield", TargetMode.MAX, metadata=metadata or MeasurableMetadata()
    )


def make_objective(metadata: TMetadata = None) -> SingleTargetObjective:
    return SingleTargetObjective(
        target=make_target(), metadata=metadata or MeasurableMetadata()
    )


@pytest.mark.parametrize(
    ("constructor", "metadata_cls"),
    [
        param(make_parameter, MeasurableMetadata, id="parameter"),
        param(make_target, MeasurableMetadata, id="target"),
        param(make_objective, Metadata, id="objective"),
    ],
)
class TestMetadataIntegration:
    """Tests for metadata integration with BayBE objects."""

    @pytest.mark.parametrize("as_dict", [True, False])
    def test_with_metadata(self, constructor, metadata_cls, as_dict: bool):
        """BayBE objects accept, ingest, and surface metadata."""
        meta = dict(description="test", unit="m", other="value")
        container = constructor(
            metadata=meta if as_dict else to_metadata(meta, metadata_cls)
        )

        assert container.description == "test"
        if metadata_cls is MeasurableMetadata:
            assert container.metadata.misc == {"other": "value"}
            assert container.unit == "m"
        else:
            assert container.metadata.misc == {"unit": "m", "other": "value"}

    def test_without_metadata(self, metadata_cls, constructor):
        """BayBE objects without metadata have empty metadata and `None` properties."""
        container = constructor().metadata

        assert container.is_empty
        assert container.description is None
        if metadata_cls is MeasurableMetadata:
            assert container.unit is None
