"""Tests for fingerprint generation."""

import pytest

from baybe._optional.info import CHEM_INSTALLED
from baybe.parameters.substance import SubstanceEncoding

test_lst = [
    (enc.name, {}, {})
    for enc in SubstanceEncoding
    if enc is not SubstanceEncoding.MORGAN_FP  # excluded due to deprecation
]


@pytest.mark.skipif(
    not CHEM_INSTALLED, reason="Optional chem dependency not installed."
)
@pytest.mark.parametrize(
    "name,kw_fp,kw_conf",
    test_lst
    + [
        ("ECFP", {"fp_size": 64}, {}),
        ("ECFP", {"fp_size": 512}, {}),
        ("ECFP", {"radius": 4}, {}),
        ("ECFP", {"fp_size": 512, "radius": 4}, {}),
        ("ECFP", {}, {"max_gen_attempts": 5000}),
    ],
)
def test_fingerprint_kwargs(name, kw_fp, kw_conf):
    """Test all fingerprint computations."""
    from baybe.utils.chemistry import smiles_to_fingerprint_features

    smiles_list = ["CC(N(C)C)=O", "CCCC#N"]
    x = smiles_to_fingerprint_features(
        smiles_list=smiles_list,
        fingerprint_name=name,
        prefix="",
        kwargs_conformer=kw_conf,
        kwargs_fingerprint=kw_fp,
    )
    # Check that fingerprint embedding is of correct size and
    # fingerprint kwargs specifying embedding size are used
    assert x.shape[0] == len(smiles_list)
    if "fp_size" in kw_fp:
        assert x.shape[1] == kw_fp["fp_size"]
