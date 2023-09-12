# pylint: disable=missing-module-docstring, missing-function-docstring
"""Test serialization of parameters."""

import pytest

from baybe.parameters import Parameter


@pytest.mark.parametrize(
    "parameter_names",
    ["Categorical_1", "Num_disc_1", "Custom_1", "Solvent_1", "Conti_finite1"],
)
@pytest.mark.parametrize("n_grid_points", [5])
def test_parameter_serialization(parameters):
    param = parameters[0]
    string = param.to_json()
    param2 = Parameter.from_json(string)
    assert param == param2
