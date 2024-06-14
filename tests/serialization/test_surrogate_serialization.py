"""Test serialization of surrogates."""

import pytest

from baybe._optional.info import ONNX_INSTALLED
from baybe.surrogates.base import Surrogate
from baybe.surrogates.custom import CustomONNXSurrogate
from baybe.utils.basic import get_subclasses


@pytest.mark.parametrize("surrogate_cls", get_subclasses(Surrogate))
def test_surrogate_serialization(request, surrogate_cls):
    """A serialization roundtrip yields an equivalent object."""
    if issubclass(surrogate_cls, CustomONNXSurrogate):
        if not ONNX_INSTALLED:
            pytest.skip("Optional onnx dependency not installed.")
        surrogate = request.getfixturevalue("onnx_surrogate")
    else:
        surrogate = surrogate_cls()

    string = surrogate.to_json()
    surrogate2 = Surrogate.from_json(string)
    assert surrogate == surrogate2, (surrogate, surrogate2)
