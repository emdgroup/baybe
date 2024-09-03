"""Chemistry tools."""

import os
import ssl
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd
from joblib import Memory

from baybe._optional.chem import (
    BaseFingerprintTransformer,
    Chem,
)

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


def smiles_to_fingerprint_features(
    smiles_list: list[str],
    fingerprint_encoder: BaseFingerprintTransformer,
    prefix: str = "",
) -> pd.DataFrame:
    """Compute molecule fingerprints for a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings.
        fingerprint_encoder: Object used to transform smiles to fingerprints
        prefix: Name prefix for each descriptor (e.g., nBase --> <prefix>_nBase).

    Returns:
        Dataframe containing fingerprints for each SMILES string.
    """
    features = fingerprint_encoder.transform(smiles_list)
    col_names = [
        prefix + "SKFP_" + f for f in fingerprint_encoder.get_feature_names_out()
    ]
    df = pd.DataFrame(features, columns=col_names)

    return df


def get_canonical_smiles(smiles: str) -> str:
    """Return the "canonical" representation of the given SMILES."""
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except Exception:
        raise ValueError(f"The SMILES '{smiles}' does not appear to be valid.")
