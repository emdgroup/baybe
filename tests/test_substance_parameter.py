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


@pytest.mark.skipif(
    not CHEM_INSTALLED, reason="Optional chem dependency not installed."
)
def test_degenerate_comp_df():
    """Test a degenerate comp_df is detected and fixed numerically."""
    from baybe.parameters import SubstanceParameter

    # These molecules are know to cause a degenerate representation with rdkit fp's
    dict_base = {
        "Potassium acetate": r"O=C([O-])C.[K+]",
        "Potassium pivalate": r"O=C([O-])C(C)(C)C.[K+]",
        "Cesium acetate": r"O=C([O-])C.[Cs+]",
        "Cesium pivalate": r"O=C([O-])C(C)(C)C.[Cs+]",
    }
    p = SubstanceParameter(name="p", data=dict_base, encoding="RDKITFINGERPRINT")

    assert (
        not p.comp_df.duplicated().any()
    ), "A degenerate comp_df was not correctly treated."
