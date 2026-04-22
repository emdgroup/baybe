"""Test alternative ways of creation not considered in the strategies."""

import pytest

from baybe.acquisition.base import AcquisitionFunction
from baybe.recommenders import BotorchRecommender
from baybe.utils.basic import get_subclasses

abbreviations = [
    cl.abbreviation
    for cl in get_subclasses(AcquisitionFunction)
    if hasattr(cl, "abbreviation")
]
fullnames = [cl.__name__ for cl in get_subclasses(AcquisitionFunction)]
combined = set(abbreviations + fullnames)


@pytest.mark.parametrize("acqf", combined)
def test_creation_from_string(acqf):
    """Tests the creation from strings."""
    AcquisitionFunction.from_dict({"type": acqf})


@pytest.mark.parametrize("acqf", combined)
def test_string_usage_in_recommender(acqf):
    """Tests the recommender initialization with acqfs as string."""
    BotorchRecommender(acquisition_function=acqf)
