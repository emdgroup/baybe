# pylint: disable=missing-module-docstring, missing-function-docstring

import pytest

from baybe.parameters import Parameter


@pytest.mark.parametrize("parameter_names", ["Categorical_1"])
def test_parameter_serialization(parameters):
    param = parameters[0]
    string = param.to_json()
    param2 = Parameter.from_json(string)
    assert param == param2
