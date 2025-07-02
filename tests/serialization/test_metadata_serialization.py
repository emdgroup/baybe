"""Test serialization of metadata."""

from hypothesis import given

from baybe.utils.metadata import Metadata

from ..hypothesis_strategies.metadata import metadata


@given(metadata())
def test_metadata_roundtrip(meta: Metadata):
    """A serialization roundtrip yields an equivalent object."""
    string = meta.to_json()
    meta2 = Metadata.from_json(string)
    assert meta == meta2, (meta, meta2)
