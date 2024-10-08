"""Chemistry tools."""

import os
import ssl
import tempfile
import urllib.request
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Memory

from baybe._optional.chem import (
    BaseFingerprintTransformer,
    Chem,
    ConformerGenerator,
    MolFromSmilesTransformer,
    skfp_fingerprints,
)
from baybe.parameters.enum import fingerprint_name_map
from baybe.utils.numerical import DTypeFloatNumpy

# Caching
_cachedir = os.environ.get(
    "BAYBE_CACHE_DIR", str(Path(tempfile.gettempdir()) / ".baybe_cache")
)


def _dummy_wrapper(func):
    return func


_disk_cache = _dummy_wrapper if _cachedir == "" else Memory(Path(_cachedir)).cache


def name_to_smiles(name: str) -> str:
    """Convert from chemical name to SMILES string using chemical identifier resolver.

    This script is useful to combine with ``df.apply`` from pandas, hence it does not
    throw exceptions for invalid molecules but instead returns an empty string for
    easy subsequent postprocessing of the dataframe.

    Args:
        name: Name or nickname of compound.

    Returns:
        SMILES string corresponding to chemical name.
    """
    name = name.replace(" ", "%20")

    try:
        url = "http://cactus.nci.nih.gov/chemical/structure/" + name + "/smiles"
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(url, context=ctx) as web:
            smiles = web.read().decode("utf8")

        smiles = str(smiles)
        if "</div>" in smiles:
            return ""

        return smiles

    except Exception:
        return ""


@lru_cache(maxsize=None)
@_disk_cache
def _molecule_to_fingerprint_features(
    fingerprint_encoder: BaseFingerprintTransformer,
    molecule: str | Chem.PropertyMol.PropertyMol,
) -> np.ndarray:
    """Compute molecular fingerprint for a single SMILES string.

    Args:
        fingerprint_encoder: Instance of Fingerprint class used to
            transform smiles string to fingerprint
        molecule: Smiles string or molecule object,
            depending on what should be input into fingerprint_encoder's transform

    Returns:
        Array containing fingerprint for SMILES string.
    """
    return fingerprint_encoder.transform([molecule])


def smiles_to_fingerprint_features(
    smiles_list: list[str],
    fingerprint_name: str,
    prefix: str = "",
    kwargs_conformer: dict | None = None,
    kwargs_fingerprint: dict | None = None,
) -> pd.DataFrame:
    """Compute molecular fingerprints for a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings.
        fingerprint_name: Name of Fingerprint class used to
            transform smiles to fingerprints
        prefix: Name prefix for each descriptor (e.g., nBase --> <prefix>_nBase).
        kwargs_conformer: kwargs for conformer generator
        kwargs_fingerprint: kwargs for fingerprint generator

    Returns:
        Dataframe containing fingerprints for each SMILES string.
    """
    fingerprint_cls, kwargs_fingerprint = convert_fingeprint_parameters(
        name=fingerprint_name, kwargs_fingerprint=kwargs_fingerprint
    )
    kwargs_conformer = kwargs_conformer or {}

    fingerprint_encoder = getattr(skfp_fingerprints, fingerprint_cls)(
        **kwargs_fingerprint
    )

    if fingerprint_encoder.requires_conformers:
        mol_list = ConformerGenerator(**kwargs_conformer).transform(
            MolFromSmilesTransformer().transform(smiles_list)
        )
    else:
        mol_list = smiles_list

    features = np.concatenate(
        [
            _molecule_to_fingerprint_features(
                fingerprint_encoder=fingerprint_encoder, molecule=mol
            )
            for mol in mol_list
        ]
    )
    name = f"skfp{fingerprint_encoder.__class__.__name__.replace('Fingerprint', '')}_"
    col_names = [prefix + name + f for f in fingerprint_encoder.get_feature_names_out()]
    df = pd.DataFrame(features, columns=col_names, dtype=DTypeFloatNumpy)

    return df


def convert_fingeprint_parameters(
    name: str, kwargs_fingerprint: dict | None = None
) -> tuple[str, dict]:
    """Convert fingerprint name parameters for computing the fingerprint.

    Args:
        name: Name of fingerprint.
        kwargs_fingerprint: Optional user-specified params
            for computing the fingerprint.

    Raises:
        KeyError: If fingerprint name is not recognized.

    Returns:
        Fingerprint class name and kwargs to use for the fingerprint computation.
    """
    # Get fingerprint class
    try:
        fp_class = fingerprint_name_map[name]
    except KeyError:
        raise KeyError(f"Fingerprint name {name} is not valid.")

    # For backwards-compatibility purposes

    # Update default kwargs to match the fingerprint name when
    # using a different fingerprint class to compute the desired fingerprint
    kwargs_fp_update = {}
    kwargs_fingerprint = {} if not kwargs_fingerprint else kwargs_fingerprint
    if name == "MORGAN_FP":
        warnings.warn(
            "Substance encoding 'MORGAN_FP' is deprecated and will be disabled in "
            "a future version. Use 'ECFP' with 'fp_size' 1204 and 'radius' 4 instead.",
            DeprecationWarning,
        )
        kwargs_fp_update = {
            "fp_size": 1024,
            "radius": 4,
        }
    # Update kwargs with fingerprint-specific defaults
    # If a kwarg is specified in the input it overrides the fingerprint default
    kwargs_fingerprint = {**kwargs_fp_update, **kwargs_fingerprint}

    return fp_class, kwargs_fingerprint


def get_canonical_smiles(smiles: str) -> str:
    """Return the "canonical" representation of the given SMILES."""
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except Exception:
        raise ValueError(f"The SMILES '{smiles}' does not appear to be valid.")
