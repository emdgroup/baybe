"""Test for fingerprint generation."""

from baybe.parameters import SubstanceEncoding
from baybe.utils.chemistry import smiles_to_fingerprint_features


def test_fingerprint_computation():
    smiles_list = ["CC(N(C)C)=O", "CCCC#N"]
    for fingerprint in SubstanceEncoding:
        smiles_to_fingerprint_features(
            smiles_list=smiles_list,
            fingerprint_name=fingerprint.name,
            prefix="",
            # Some params that make the test faster
            kwargs_conformer={
                "max_gen_attempts": 5000,
                "n_jobs": 4,
            },
            kwargs_fingerprint={
                "n_jobs": 4,
            },
        )

    # Also run one time without passing kwargs
    smiles_to_fingerprint_features(
        smiles_list=smiles_list,
        fingerprint_name=SubstanceEncoding("MORGAN_FP").name,
        prefix="",
        kwargs_conformer=None,
        kwargs_fingerprint=None,
    )
