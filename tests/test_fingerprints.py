"""Tests for fingerprint generation."""

import pytest

from baybe._optional.info import CHEM_INSTALLED
from baybe.parameters.enum import SubstanceEncoding

test_lst = [
    (enc, {}, {})
    for enc in SubstanceEncoding
    if enc
    not in {  # Ignore deprecated encodings
        SubstanceEncoding.MORGAN_FP,
        SubstanceEncoding.RDKIT,
    }
]

ECFP = SubstanceEncoding.ECFP


@pytest.mark.skipif(
    not CHEM_INSTALLED, reason="Optional chem dependency not installed."
)
@pytest.mark.parametrize(
    "encoding,kw_fp,kw_conf",
    test_lst
    + [  # Add some custom tests
        (ECFP, {"fp_size": 64}, {}),
        (ECFP, {"fp_size": 512}, {}),
        (ECFP, {"radius": 4}, {}),
        (ECFP, {"fp_size": 512, "radius": 4}, {}),
        (ECFP, {}, {"max_gen_attempts": 5000}),
    ],
)
def test_fingerprint_kwargs(encoding, kw_fp, kw_conf):
    """Test all fingerprint computations."""
    from baybe.utils.chemistry import smiles_to_fingerprint_features

    smiles_list = ["CC(N(C)C)=O", "CCCC#N"]
    x = smiles_to_fingerprint_features(
        smiles_list=smiles_list,
        encoding=encoding,
        prefix="",
        kwargs_conformer=kw_conf,
        kwargs_fingerprint=kw_fp,
    )

    # Check that fingerprint embedding is of correct size and
    # fingerprint kwargs specifying embedding size are used
    assert x.shape[0] == len(smiles_list)
    if "fp_size" in kw_fp:
        assert x.shape[1] == kw_fp["fp_size"]
