"""Test alternative ways of creation not considered in the strategies."""

import pytest

from baybe._optional.info import CHEM_INSTALLED
from baybe.parameters.categorical import CategoricalParameter
from baybe.parameters.enum import CategoricalEncoding, SubstanceEncoding
from baybe.parameters.substance import SubstanceParameter


@pytest.mark.parametrize("encoding", [e.name for e in CategoricalEncoding])
def test_string_encoding_categorical_parameter(encoding):
    """The encoding can also be specified as a string instead of an enum value."""
    CategoricalParameter(name="string_encoding", values=["A", "B"], encoding=encoding)


@pytest.mark.skipif(
    not CHEM_INSTALLED, reason="Optional chem dependency not installed."
)
@pytest.mark.parametrize("encoding", [e.name for e in SubstanceEncoding])
def test_string_encoding_substance_parameter(encoding):
    """The encoding can also be specified as a string instead of an enum value."""
    SubstanceParameter(
        name="string_encoding", data={"A": "C", "B": "CC"}, encoding=encoding
    )
