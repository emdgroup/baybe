"""Tests for the substance parameter."""

import pytest

from baybe.utils.chemistry import _MORDRED_INSTALLED, _RDKIT_INSTALLED

from .conftest import run_iterations

_CHEM_INSTALLED = _MORDRED_INSTALLED and _RDKIT_INSTALLED
if _CHEM_INSTALLED:
    from baybe.parameters import SUBSTANCE_ENCODINGS


if _CHEM_INSTALLED:

    @pytest.mark.parametrize(
        "parameter_names",
        [["Categorical_1", f"Substance_1_{enc}"] for enc in SUBSTANCE_ENCODINGS],
        ids=SUBSTANCE_ENCODINGS,
    )
    def test_run_iterations(baybe, batch_quantity, n_iterations):
        """Test running some iterations with fake results and a substance parameter."""
        run_iterations(baybe, n_iterations, batch_quantity)
