"""Test serialization of surrogates."""

import pytest

from baybe.surrogates.base import Surrogate, get_available_surrogates


@pytest.mark.parametrize("surrogate", [cls() for cls in get_available_surrogates()])
def test_surrogate_serialization(surrogate):
    string = surrogate.to_json()
    surrogate2 = Surrogate.from_json(string)
    assert surrogate == surrogate2
