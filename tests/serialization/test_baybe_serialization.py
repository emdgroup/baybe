# pylint: disable=missing-module-docstring, missing-function-docstring
"""Test sserialization of BayBE objects."""

import pytest

from baybe.core import Campaign
from cattrs import ClassValidationError


def roundtrip(baybe: Campaign) -> Campaign:
    string = baybe.to_json()
    return Campaign.from_json(string)


def test_baybe_serialization(baybe):

    baybe2 = roundtrip(baybe)
    assert baybe == baybe2

    baybe.recommend()
    baybe2 = roundtrip(baybe)
    assert baybe == baybe2


def test_valid_config(config):
    Campaign.validate_config(config)


def test_invalid_config(config):
    config = config.replace("CategoricalParameter", "CatParam")
    with pytest.raises(ClassValidationError):
        Campaign.validate_config(config)
