"""Deserialization tests.

The purpose of these tests is to ensure that deserialization works for all possible ways
in which type information can be provided.

NOTE: The tests are based on `AcquisitionFunction` simply because the class provides the
    `abbreviation` variable that is necessary for the tests.
"""

import pytest
from cattrs.errors import ClassValidationError
from pytest import param

from baybe.acquisition.acqfs import UpperConfidenceBound
from baybe.acquisition.base import AcquisitionFunction


@pytest.mark.parametrize(
    ("cls", "dct"),
    [
        param(
            AcquisitionFunction,
            {"type": "UpperConfidenceBound", "beta": 1337},
            id="base-full",
        ),
        param(
            AcquisitionFunction,
            {"type": "UCB", "beta": 1337},
            id="base-abbreviation",
        ),
        param(
            UpperConfidenceBound,
            {"beta": 1337},
            id="subclass",
        ),
    ],
)
def test_valid_deserialization(cls, dct):
    """Serialization is possible from concrete class and from base class.

    Concrete class works with: full type name, type abbreviation, and without type.
    Base class works with: full type name and type abbreviation.
    """
    UpperConfidenceBound(beta=1337) == cls.from_dict(dct)


def test_invalid_deserialization_missing_type():
    """Omitting necessary type information throws the correct error."""
    dct = {"beta": 1337}
    with pytest.raises(KeyError, match="type"):
        AcquisitionFunction.from_dict(dct)


def test_invalid_deserialization_unneeded_type():
    """Providing unneeded type information throws the correct error."""
    dct = {"type": "UCB", "beta": 1337}
    with pytest.raises(ClassValidationError) as ex:
        UpperConfidenceBound.from_dict(dct)
    assert len(ex.value.exceptions) == 1
    assert (
        str(ex.value.exceptions[0])
        == "Extra fields in constructor for UpperConfidenceBound: type"
    )
