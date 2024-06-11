"""Tests for the substance parameter."""

import pytest

from baybe._optional.info import CHEM_INSTALLED
from baybe.parameters.enum import SubstanceEncoding

from .conftest import run_iterations


@pytest.mark.skipif(
    not CHEM_INSTALLED, reason="Optional chem dependency not installed."
)
@pytest.mark.parametrize(
    "parameter_names",
    [["Categorical_1", f"Substance_1_{enc}"] for enc in SubstanceEncoding],
    ids=[enc.name for enc in SubstanceEncoding],
)
def test_run_iterations(campaign, batch_size, n_iterations):
    """Test running some iterations with fake results and a substance parameter."""
    run_iterations(campaign, n_iterations, batch_size)
