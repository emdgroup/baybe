"""Symmetry serialization tests."""

import hypothesis.strategies as st
import pytest
from hypothesis import given
from pytest import param

from tests.hypothesis_strategies.symmetry import (
    dependency_symmetries,
    mirror_symmetries,
    permutation_symmetries,
)
from tests.serialization.utils import assert_roundtrip_consistency


@pytest.mark.parametrize(
    "strategy",
    [
        param(mirror_symmetries(), id="MirrorSymmetry"),
        param(permutation_symmetries(), id="PermutationSymmetry"),
        param(dependency_symmetries(), id="DependencySymmetry"),
    ],
)
@given(data=st.data())
def test_roundtrip(strategy: st.SearchStrategy, data: st.DataObject):
    """A serialization roundtrip yields an equivalent object."""
    symmetry = data.draw(strategy)
    assert_roundtrip_consistency(symmetry)
