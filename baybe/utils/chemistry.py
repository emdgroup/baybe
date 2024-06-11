"""Chemistry tools."""

import os
import ssl
import tempfile
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Memory

from baybe._optional.chem import (
    Calculator,
    Chem,
    GetMorganFingerprintAsBitVect,
    RDLogger,
    descriptors,
)
from baybe.utils.numerical import DTypeFloatNumpy

_mordred_calculator = Calculator(descriptors)


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
def _smiles_to_mordred_features(smiles: str) -> np.ndarray:
    """Memory- and disk-cached computation of Mordred descriptors.

    Args:
        smiles: SMILES string.

    Returns:
        Mordred descriptors for the given smiles string.
    """
    try:
        return np.asarray(
            _mordred_calculator(Chem.MolFromSmiles(smiles)).fill_missing()
        )
    except Exception:
        return np.full(len(_mordred_calculator.descriptors), np.nan)


def smiles_to_mordred_features(
    smiles_list: list[str],
    prefix: str = "",
    dropna: bool = True,
) -> pd.DataFrame:
    """Compute Mordred chemical descriptors for a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings.
        prefix: Name prefix for each descriptor
            (e.g., nBase --> <prefix>_nBase).
        dropna: If ``True``, drops columns that contain NaNs.

    Returns:
        Dataframe containing overlapping Mordred descriptors for each SMILES
        string.
    """
    features = [_smiles_to_mordred_features(smiles) for smiles in smiles_list]
    descriptor_names = list(_mordred_calculator.descriptors)
    columns = [prefix + "MORDRED_" + str(name) for name in descriptor_names]
    dataframe = pd.DataFrame(data=features, columns=columns, dtype=DTypeFloatNumpy)

    if dropna:
        dataframe = dataframe.dropna(axis=1)

    return dataframe


def smiles_to_molecules(smiles_list: list[str]) -> list[Chem.Mol]:
    """Convert a given list of SMILES strings into corresponding Molecule objects.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        List of corresponding molecules.

    Raises:
        ValueError: If the SMILES does not seem to be chemically valid.
    """
    mols = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError()
            mols.append(mol)
        except Exception as ex:
            raise ValueError(
                f"The SMILES {smiles} does not seem to be chemically valid."
            ) from ex
    return mols


def smiles_to_rdkit_features(
    smiles_list: list[str], prefix: str = "", dropna: bool = True
) -> pd.DataFrame:
    """Compute RDKit chemical descriptors for a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings.
        prefix: Name prefix for each descriptor (e.g., nBase --> <prefix>_nBase).
        dropna: If ``True``, drops columns that contain NaNs.

    Returns:
        Dataframe containing overlapping RDKit descriptors for each SMILES string.
    """
    mols = smiles_to_molecules(smiles_list)

    res = []
    for mol in mols:
        desc = {
            prefix + "RDKIT_" + dname: DTypeFloatNumpy(func(mol))
            for dname, func in Chem.Descriptors.descList
        }
        res.append(desc)

    df = pd.DataFrame(res)
    if dropna:
        df = df.dropna(axis=1)

    return df


def smiles_to_fp_features(
    smiles_list: list[str],
    prefix: str = "",
    dtype: type[int] | type[float] = int,
    radius: int = 4,
    n_bits: int = 1024,
) -> pd.DataFrame:
    """Compute standard Morgan molecule fingerprints for a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings.
        prefix: Name prefix for each descriptor (e.g., nBase --> <prefix>_nBase).
        dtype: Specifies whether fingerprints will have int or float data type.
        radius: Radius for the Morgan fingerprint.
        n_bits:Number of bits for the Morgan fingerprint.

    Returns:
        Dataframe containing Morgan fingerprints for each SMILES string.
    """
    mols = smiles_to_molecules(smiles_list)

    res = []
    for mol in mols:
        RDLogger.logger().setLevel(RDLogger.CRITICAL)

        fingerp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits).ToBitString()
        fingerp = map(int, fingerp)
        fpvec = np.array(list(fingerp))
        res.append(
            {prefix + "FP_" + f"{k + 1}": dtype(bit) for k, bit in enumerate(fpvec)}
        )

    df = pd.DataFrame(res)

    return df


def is_valid_smiles(smiles: str) -> bool:
    """Test if a SMILES string is valid according to RDKit.

    Args:
        smiles: SMILES string to be tested.

    Returns:
        ``True`` if the provided SMILES is valid, ``False`` else.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def get_canonical_smiles(smiles: str) -> str:
    """Return the "canonical" representation of the given SMILES."""
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except Exception:
        raise ValueError(f"The SMILES '{smiles}' does not appear to be valid.")
