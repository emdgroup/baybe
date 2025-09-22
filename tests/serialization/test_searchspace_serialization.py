"""Search space serialization tests."""

import json
from collections.abc import Sequence

import pytest

from baybe.parameters.base import Parameter
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.searchspace.discrete import SubspaceDiscrete
from baybe.serialization.core import converter
from tests.serialization.utils import assert_roundtrip_consistency


@pytest.mark.parametrize(
    "parameter_names",
    [
        ["Categorical_1", "Num_disc_1"],
        ["Fraction_1"],
        ["Conti_finite1"],
        ["Custom_1"],
        ["Solvent_1"],
    ],
)
def test_roundtrip(parameters: Sequence[Parameter]) -> None:
    """A serialization roundtrip yields an equivalent object."""
    searchspace = SearchSpace.from_product(parameters)
    assert_roundtrip_consistency(searchspace)


def test_from_dataframe_deserialization(searchspace):
    """Deserialization from dataframe yields back the original object."""
    unstructured = searchspace.discrete.to_dict()
    df_string = json.dumps(unstructured["exp_rep"])
    parameters_string = json.dumps(unstructured["parameters"])
    config = """
    {
        "constructor": "from_dataframe",
        "df": __fillin_dataframe__,
        "parameters": __fillin__parameters__
    }
    """.replace("__fillin_dataframe__", df_string).replace(
        "__fillin__parameters__", parameters_string
    )
    deserialized = SearchSpace.from_json(config)
    assert searchspace == deserialized, (searchspace, deserialized)


def test_from_simplex_deserialization():
    """Deserialization from simplex yields back the original object."""
    parameters = [
        NumericalDiscreteParameter("p1", [0, 0.5, 1]),
        NumericalDiscreteParameter("p2", [0, 0.5, 1]),
    ]
    max_sum = 1.0
    subspace = SubspaceDiscrete.from_simplex(max_sum, simplex_parameters=parameters)
    parameters_string = json.dumps(converter.unstructure(parameters))
    config = """
    {
        "constructor": "from_simplex",
        "max_sum": __fillin__max_sum__,
        "simplex_parameters": __fillin__parameters__
    }
    """.replace("__fillin__max_sum__", str(max_sum)).replace(
        "__fillin__parameters__", parameters_string
    )
    deserialized = SubspaceDiscrete.from_json(config)
    assert subspace == deserialized, (subspace, deserialized)
