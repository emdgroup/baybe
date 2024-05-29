"""Test alternative ways of creation not considered in the strategies."""

import pytest

from baybe.acquisition.base import AcquisitionFunction
from baybe.recommenders import BotorchRecommender
from baybe.utils.basic import get_subclasses

abbreviation_list = [
    cl.abbreviation
    for cl in get_subclasses(AcquisitionFunction)
    if hasattr(cl, "abbreviation")
]

fullname_list = [cl.__name__ for cl in get_subclasses(AcquisitionFunction)]


@pytest.mark.parametrize("acqf", abbreviation_list + fullname_list)
def test_creation_from_string(acqf):
    """Tests the creation from strings."""
    AcquisitionFunction.from_dict({"type": acqf})


@pytest.mark.parametrize("acqf", abbreviation_list + fullname_list)
def test_string_usage_in_recommender(acqf):
    """Tests the recommender initialization with acqfs as string."""
    BotorchRecommender(acquisition_function=acqf)
