# pylint: disable=missing-module-docstring, missing-function-docstring

from baybe.searchspace import SearchSpace


def test_searchspace_serialization(parameters):
    searchspace = SearchSpace.create(parameters)
    string = searchspace.to_json()
    searchspace2 = SearchSpace.from_json(string)
    assert searchspace == searchspace2
