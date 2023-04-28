# pylint: disable=missing-module-docstring, missing-function-docstring

from baybe.core import BayBE


def roundtrip(baybe: BayBE) -> BayBE:
    string = baybe.to_json()
    return BayBE.from_json(string)


def test_baybe_serialization(baybe):

    baybe2 = roundtrip(baybe)
    assert baybe == baybe2

    baybe.recommend()
    baybe2 = roundtrip(baybe)
    assert baybe == baybe2
