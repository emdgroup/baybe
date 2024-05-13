"""Test serialization of campaigns."""

import pytest
from cattrs import ClassValidationError

from baybe.campaign import Campaign


def roundtrip(campaign: Campaign) -> Campaign:
    string = campaign.to_json()
    return Campaign.from_json(string)


def test_campaign_serialization(campaign):
    campaign2 = roundtrip(campaign)
    assert campaign == campaign2

    campaign.recommend(batch_size=1)
    campaign2 = roundtrip(campaign)
    assert campaign == campaign2


def test_valid_product_config(config):
    Campaign.validate_config(config)


def test_invalid_product_config(config):
    config = config.replace("CategoricalParameter", "CatParam")
    with pytest.raises(ClassValidationError):
        Campaign.validate_config(config)


def test_valid_simplex_config(simplex_config):
    Campaign.validate_config(simplex_config)


def test_invalid_simplex_config(simplex_config):
    simplex_config = simplex_config.replace("0.0, ", "-1.0, 0.0, ")
    with pytest.raises(ClassValidationError):
        Campaign.validate_config(simplex_config)
