# pylint: disable=missing-module-docstring, missing-function-docstring
import pytest

from baybe.core import BayBE
from cattrs import ClassValidationError


def roundtrip(baybe: BayBE) -> BayBE:
    string = baybe.to_json()
    return BayBE.from_json(string)


def test_baybe_serialization(baybe):

    baybe2 = roundtrip(baybe)
    assert baybe == baybe2

    baybe.recommend()
    baybe2 = roundtrip(baybe)
    assert baybe == baybe2


def test_valid_config(config):
    BayBE.validate_config(config)


def test_invalid_config(config):
    config = config.replace("CategoricalParameter", "CatParam")
    with pytest.raises(ClassValidationError):
        BayBE.validate_config(config)
