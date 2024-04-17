"""Test serialization of surrogates."""

import pytest

from baybe.surrogates.base import Surrogate
from baybe.utils.basic import get_subclasses

# TODO: Add serialization test for ONNX surrogate
surrogates = [
    cls() for cls in get_subclasses(Surrogate) if cls.__name__ != "CustomONNXSurrogate"
]


@pytest.mark.parametrize("surrogate", surrogates)
def test_surrogate_serialization(surrogate):
    string = surrogate.to_json()
    surrogate2 = Surrogate.from_json(string)
    assert surrogate == surrogate2
