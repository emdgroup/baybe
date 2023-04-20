# pylint: disable=missing-module-docstring, missing-function-docstring

import json

from baybe.core import BayBE


def test_serialization(baybe):

    string = json.dumps(baybe.to_dict())
    baybe2 = BayBE.from_dict(json.loads(string))
    assert baybe == baybe2
