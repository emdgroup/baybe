"""Test serialization of metadata."""

import hypothesis.strategies as st
import pytest
from hypothesis import given
from pytest import param

from baybe.utils.metadata import Metadata, to_metadata
from tests.hypothesis_strategies.metadata import metadata, parameter_metadata


@pytest.mark.parametrize(
    "metadata_strategy",
    [
        param(metadata(), id="Metadata"),
        param(parameter_metadata(), id="ParameterMetadata"),
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


def test_field_separation():
    """Field separation works regardless of the conversion route."""
    dct = {"description": "test", "unit": "m", "key": "value"}
    metadata = Metadata("test", "m", {"key": "value"})
    via_converter = to_metadata(dct)
    via_from_dict = Metadata.from_dict(dct)
    assert metadata == via_converter == via_from_dict
