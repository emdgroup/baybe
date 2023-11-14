"""Tests for the substance parameter."""

import pytest

from baybe.parameters.enum import SubstanceEncoding
from baybe.utils.chemistry import _MORDRED_INSTALLED, _RDKIT_INSTALLED

from .conftest import run_iterations

_CHEM_INSTALLED = _MORDRED_INSTALLED and _RDKIT_INSTALLED


if _CHEM_INSTALLED:

    @pytest.mark.parametrize(
        "parameter_names",
        [["Categorical_1", f"Substance_1_{enc}"] for enc in SubstanceEncoding],
        ids=[enc.name for enc in SubstanceEncoding],
    )
    def test_run_iterations(campaign, batch_quantity, n_iterations):
        """Test running some iterations with fake results and a substance parameter."""
        run_iterations(campaign, n_iterations, batch_quantity)
