"""Test for fingerprint generation."""

import pytest

from baybe.utils.chemistry import smiles_to_fingerprint_features


@pytest.mark.parametrize(
    "fingerprint_name,kwargs_fingerprint,kwargs_conformer",
    [
        # Test fingerprint calculation with different kwargs
        ("ECFP", {}, {}),
        ("ECFP", {"fp_size": 64}, {}),
        ("ECFP", {}, {"max_gen_attempts": 5000}),
    ],
)
def test_fingerprint_kwargs(fingerprint_name, kwargs_fingerprint, kwargs_conformer):
    smiles_list = ["CC(N(C)C)=O", "CCCC#N"]
    x = smiles_to_fingerprint_features(
        smiles_list=smiles_list,
        fingerprint_name=fingerprint_name,
        prefix="",
        kwargs_conformer=kwargs_conformer,
        kwargs_fingerprint=kwargs_fingerprint,
    )
    # Check that fingerprint embedding is of correct size and
    # fingerprint kwargs specifying embedding size are used
    assert x.shape[0] == len(smiles_list)
    if "fp_size" in kwargs_fingerprint:
        assert x.shape[1] == kwargs_fingerprint["fp_size"]
