"""Kernel factory serialization tests."""

import pytest

from baybe.surrogates.gaussian_process.components.kernel import _PureKernelFactory
from baybe.surrogates.gaussian_process.presets import *  # noqa: F401, F403
from baybe.utils.basic import get_subclasses
from tests.serialization.utils import assert_roundtrip_consistency

_KERNEL_FACTORIES = [
    cls
    for cls in get_subclasses(_PureKernelFactory)
    if not cls.__name__.startswith("_")
]


@pytest.mark.parametrize(
    "factory",
    [pytest.param(cls(), id=cls.__name__) for cls in _KERNEL_FACTORIES],
)
def test_roundtrip(factory):
    """A serialization roundtrip yields an equivalent object."""
    assert_roundtrip_consistency(factory)
