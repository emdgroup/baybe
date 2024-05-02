"""Test serialization of surrogates."""

import pytest

from baybe.surrogates.base import Surrogate
from baybe.utils.basic import get_subclasses


@pytest.mark.parametrize("surrogate_cls", get_subclasses(Surrogate))
def test_surrogate_serialization(surrogate_cls, onnx_surrogate):
    """A serialization roundtrip yields an equivalent object."""
    if surrogate_cls.__name__ == "CustomONNXSurrogate":
        surrogate = onnx_surrogate
    else:
        surrogate = surrogate_cls()

    string = surrogate.to_json()
    surrogate2 = Surrogate.from_json(string)
    assert surrogate == surrogate2, (surrogate, surrogate2)
