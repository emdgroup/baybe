# pylint: disable=missing-module-docstring, missing-function-docstring

from baybe.parameters import Categorical, Parameter


def test_categorical():
    param = Categorical("cat", [1, 2, 3])
    param2 = Parameter.from_dict(param.to_dict())
    assert param == param2
