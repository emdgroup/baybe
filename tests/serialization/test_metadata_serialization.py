"""Test serialization of metadata."""

import hypothesis.strategies as st
import pytest
from hypothesis import given
from pytest import param

from baybe.utils.metadata import MeasurableMetadata, Metadata, to_metadata
from tests.hypothesis_strategies.metadata import measurable_metadata, metadata


@pytest.mark.parametrize(
    "metadata_strategy",
    [
        param(metadata(), id="Metadata"),
        param(measurable_metadata(), id="MeasurableMetadata"),
    ],
)
@given(data=st.data())
def test_metadata_roundtrip(data, metadata_strategy):
    """A serialization roundtrip yields an equivalent object."""
    meta = data.draw(metadata_strategy)
    cls = type(meta)
    string = meta.to_json()
    meta2 = cls.from_json(string)
    assert meta == meta2, (meta, meta2)


@pytest.mark.parametrize("cls", [Metadata, MeasurableMetadata])
def test_field_separation(cls: type[Metadata]):
    """Field separation adapts to the specific ``Metadata`` subclass and works
    regardless of the conversion route.
    """  # noqa: D205
    dct = {"description": "test", "unit": "m", "key": "value"}
    if cls is Metadata:
        metadata = cls(description="test", misc={"unit": "m", "key": "value"})
    elif issubclass(cls, MeasurableMetadata):
        metadata = cls(description="test", unit="m", misc={"key": "value"})
    else:
        raise ValueError(f"Unsupported class: {cls}")
    via_converter = to_metadata(dct, cls)
    via_from_dict = cls.from_dict(dct)
    assert metadata == via_converter == via_from_dict
