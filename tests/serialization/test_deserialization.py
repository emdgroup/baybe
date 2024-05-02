"""Deserialization tests.

The purpose of these tests is to ensure that deserialization works for all possible ways
in which type information can be provided.

NOTE: The tests are based on `AcquisitionFunction` simply because the class provides the
    `abbreviation` variable that is necessary for the tests.
"""

import pytest
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
            {"type": "UpperConfidenceBound", "beta": 1337},
            id="subclass-full",
        ),
        param(
            UpperConfidenceBound,
            {"type": "UCB", "beta": 1337},
            id="subclass-abbreviation",
        ),
        param(
            UpperConfidenceBound,
            {"beta": 1337},
            id="subclass-without",
        ),
    ],
)
def test_valid_deserialization(cls, dct):
    """Serialization is possible from concrete class and from base class.

    Concrete class works with: full type name, type abbreviation, and without type.
    Base class works with: full type name and type abbreviation.
    """
    UpperConfidenceBound(beta=1337) == cls.from_dict(dct)


@pytest.mark.parametrize(
    ("cls", "dct", "err", "msg"),
    [
        param(
            AcquisitionFunction,
            {"beta": 1337},
            KeyError,
            "type",
            id="missing-type",
        ),
        param(
            UpperConfidenceBound,
            {"type": "wrong", "beta": 1337},
            ValueError,
            "'UpperConfidenceBound' .* does not match .* 'wrong'",
            id="inconsistent-type",
        ),
    ],
)
def test_invalid_deserialization(cls, dct, err, msg):
    """Incorrect type information throws the correct error."""
    with pytest.raises(err, match=msg):
        cls.from_dict(dct)
