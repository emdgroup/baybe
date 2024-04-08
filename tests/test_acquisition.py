"""Acquisition function tests."""

import pytest
from pytest import param

from baybe.acquisition.base import AcquisitionFunction
from baybe.recommenders import SequentialGreedyRecommender
from baybe.utils.basic import get_subclasses

acqfs = [  # by object
    param(cl(), id=f"{cl.__name__}_obj") for cl in get_subclasses(AcquisitionFunction)
]
acqfs += [cl.__name__ for cl in get_subclasses(AcquisitionFunction)]  # by long name
acqfs += [  # by abbreviation
    cl._abbreviation for cl in get_subclasses(AcquisitionFunction)
]


@pytest.mark.parametrize("acqf", acqfs)
def test_deprecated_acqfs2(acqf):
    """Using the deprecated strategy keyword raises an error."""
    SequentialGreedyRecommender(acqf=acqf)
