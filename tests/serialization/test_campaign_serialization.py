"""Campaign serialization tests."""

import pytest
from cattrs import ClassValidationError

from baybe.campaign import Campaign
from baybe.utils.dataframe import add_fake_measurements
from tests.serialization.utils import assert_roundtrip_consistency


def test_roundtrip(campaign: Campaign):
    """A serialization roundtrip yields an equivalent object."""
    assert_roundtrip_consistency(campaign)

    # Let's also confirm consistency after completing one DOE iteration
    recommendation = campaign.recommend(batch_size=1)
    add_fake_measurements(recommendation, campaign.targets)
    campaign.add_measurements(recommendation)
    assert_roundtrip_consistency(campaign)


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
