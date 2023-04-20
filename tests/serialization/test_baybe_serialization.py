# pylint: disable=missing-module-docstring, missing-function-docstring

import json

from baybe.core import BayBE


def roundtrip(baybe: BayBE) -> BayBE:
    string = json.dumps(baybe.to_dict())
    return BayBE.from_dict(json.loads(string))


def test_serialization(baybe):

    baybe2 = roundtrip(baybe)
    assert baybe == baybe2

    baybe.recommend()
    baybe2 = roundtrip(baybe)
    assert baybe == baybe2
