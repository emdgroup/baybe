"""Test serialization of parameters."""

from hypothesis import given

from baybe.parameters.base import Parameter

from ..hypothesis_strategies import parameter


@given(parameter)
def test_parameter_roundtrip(param: Parameter):
    """A serialization roundtrip yields an equivalent object."""
    string = param.to_json()
    param2 = Parameter.from_json(string)
    assert param == param2, (param, param2)
