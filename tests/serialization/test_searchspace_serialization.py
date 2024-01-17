"""Test serialization of searchspaces."""

import json

import pytest

from baybe.searchspace import SearchSpace


@pytest.mark.parametrize(
    "parameter_names",
    [
        ["Categorical_1", "Num_discrete_1"],
        ["Fraction_1"],
        ["Conti_finite1"],
        ["Custom_1"],
        ["Solvent_1"],
    ],
)
def test_searchspace_serialization(parameters):
    searchspace = SearchSpace.from_product(parameters)
    string = searchspace.to_json()
    searchspace2 = SearchSpace.from_json(string)
    assert searchspace == searchspace2


def test_from_dataframe_deserialization(searchspace):
    """Deserialization from dataframe yields back the original object."""
    unstructured = searchspace.discrete.to_dict()
    df_string = json.dumps(unstructured["exp_rep"])
    parameters_string = json.dumps(unstructured["parameters"])
    config = """
    {
        "constructor": "from_dataframe",
        "dataframe": __fillin_dataframe__,
        "parameters": __fillin__parameters__
    }
    """.replace("__fillin_dataframe__", df_string).replace(
        "__fillin__parameters__", parameters_string
    )
    deserialized = SearchSpace.from_json(config)
    assert searchspace == deserialized, (searchspace, deserialized)
