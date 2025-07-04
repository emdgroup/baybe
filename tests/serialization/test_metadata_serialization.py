"""Test serialization of metadata."""

from hypothesis import given

from baybe.utils.metadata import Metadata, to_metadata

from ..hypothesis_strategies.metadata import metadata


@given(metadata())
def test_metadata_roundtrip(meta: Metadata):
    """A serialization roundtrip yields an equivalent object."""
    string = meta.to_json()
    meta2 = Metadata.from_json(string)
    assert meta == meta2, (meta, meta2)


def test_field_separation():
    """Field separation works regardless of the conversion route."""
    dct = {"description": "test", "unit": "m", "key": "value"}
    metadata = Metadata("test", "m", {"key": "value"})
    via_converter = to_metadata(dct)
    via_from_dict = Metadata.from_dict(dct)
    assert metadata == via_converter == via_from_dict
