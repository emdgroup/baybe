"""Tests for fingerprint generation."""

import pytest

from baybe._optional.info import CHEM_INSTALLED
from baybe.parameters.enum import SubstanceEncoding

test_cases: list[tuple[SubstanceEncoding, dict, dict]] = [
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
    ("encoding", "kw_fp", "kw_conf"),
    test_cases
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

    smiles = ["CC(N(C)C)=O", "CCCC#N"]
    x = smiles_to_fingerprint_features(
        smiles=smiles,
        encoding=encoding,
        prefix="",
        kwargs_conformer=kw_conf,
        kwargs_fingerprint=kw_fp,
    )

    assert x.shape[0] == len(smiles), (
        "The number of fingerprint embedding rows does not match the number of "
        "molecules."
    )
    if "fp_size" in kw_fp:
        assert x.shape[1] == kw_fp["fp_size"], (
            "The fingerprint dimension parameter was ignored, fingerprints have a "
            "wrong number of dimensions."
        )
