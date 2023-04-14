# pylint: disable=missing-module-docstring, missing-function-docstring

import json

from baybe.searchspace import SearchSpace


def test_searchspace_serialization(parameters):
    searchspace = SearchSpace.create(parameters)
    string = json.dumps(searchspace.to_dict())
    searchspace2 = SearchSpace.from_dict(json.loads(string))
    assert searchspace == searchspace2
