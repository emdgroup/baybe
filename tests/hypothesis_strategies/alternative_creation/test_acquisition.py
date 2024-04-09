"""Test alternative ways of creation not considered in the strategies."""

import pytest

from baybe.acquisition.base import AcquisitionFunction
from baybe.recommenders.pure.bayesian import SequentialGreedyRecommender
from baybe.utils.basic import get_subclasses

abbreviation_list = [
    cl._abbreviation
    for cl in get_subclasses(AcquisitionFunction)
    if hasattr(cl, "_abbreviation")
]

fullname_list = [cl.__name__ for cl in get_subclasses(AcquisitionFunction)]


@pytest.mark.parametrize("acqf", abbreviation_list)
def test_string_abbreviation(acqf):
    """Tests the creation from abbreviation strings."""
    SequentialGreedyRecommender(acqf=acqf)


@pytest.mark.parametrize("acqf", fullname_list)
def test_string_fullname(acqf):
    """Tests the creation from fullname strings."""
    SequentialGreedyRecommender(acqf=acqf)
