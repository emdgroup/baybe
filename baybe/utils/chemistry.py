"""Chemistry tools."""

import ssl
import urllib.request

from functools import lru_cache
from pathlib import Path
from typing import List, Type, Union

import numpy as np
import pandas as pd
from joblib import Memory
from mordred import Calculator, descriptors
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect


# Caching
cachedir = Path.home() / ".baybe_cache"
memory_utils = Memory(cachedir / "utils")

# Fingerprint calculation
mordred_calculator = Calculator(descriptors)


def name_to_smiles(name: str) -> str:
    """
    Converts from chemical name to SMILES string using chemical identifier resolver.

    Parameters
    ----------
    name : str
        Name or nickname of compound.

    Returns
    -------
    str
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
@memory_utils.cache
def _smiles_to_mordred_features(smiles: str) -> np.ndarray:
    """
    Memory- and disk-cached computation of Mordred descriptors.

    Parameters
    ----------
    smiles : str
        SMILES string.

    Returns
    -------
    np.ndarray
        Mordred descriptors for the given smiles string.
    """
    try:
        return np.asarray(mordred_calculator(Chem.MolFromSmiles(smiles)).fill_missing())
    except Exception:
        return np.full(len(mordred_calculator.descriptors), np.NaN)


def smiles_to_mordred_features(
    smiles_list: List[str],
    prefix: str = "",
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Computes Mordred chemical descriptors for a list of SMILES strings.

    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings.
    prefix : str
        Name prefix for each descriptor (e.g., nBase --> <prefix>_nBase).
    dropna : bool
        If True, drops columns that contain NaNs.

    Returns
    -------
    pd.DataFrame
        Dataframe containing overlapping Mordred descriptors for each SMILES string.
    """
    features = [_smiles_to_mordred_features(smiles) for smiles in smiles_list]
    descriptor_names = list(mordred_calculator.descriptors)
    columns = [prefix + "MORDRED_" + str(name) for name in descriptor_names]
    dataframe = pd.DataFrame(data=features, columns=columns)

    if dropna:
        dataframe = dataframe.dropna(axis=1)

    return dataframe


def smiles_to_molecules(smiles_list: List[str]) -> List[Chem.Mol]:
    """
    Converts a given list of SMILES strings into corresponding Molecule objects.

    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings.

    Returns
    -------
    List[Chem.Mol]
        List of corresponding molecules.
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
    smiles_list: List[str], prefix: str = "", dropna: bool = True
) -> pd.DataFrame:
    """
    Computes RDKit chemical descriptors for a list of SMILES strings.

    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings.
    prefix : str
        Name prefix for each descriptor (e.g., nBase --> <prefix>_nBase).
    dropna : bool
        If True, drops columns that contain NaNs.

    Returns
    ----------
    pd.DataFrame
        Dataframe containing overlapping RDKit descriptors for each SMILES string.
    """
    mols = smiles_to_molecules(smiles_list)

    res = []
    for mol in mols:
        desc = {
            prefix + "RDKIT_" + dname: func(mol)
            for dname, func in Chem.Descriptors.descList
        }
        res.append(desc)

    df = pd.DataFrame(res)
    if dropna:
        df = df.dropna(axis=1)

    return df


def smiles_to_fp_features(
    smiles_list: List[str],
    prefix: str = "",
    dtype: Union[Type[int], Type[float]] = int,
    radius: int = 4,
    n_bits: int = 1024,
) -> pd.DataFrame:
    """
    Computes standard Morgan molecule fingerprints for a list of SMILES strings.

    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings.
    prefix : str
        Name prefix for each descriptor (e.g., nBase --> <prefix>_nBase).
    dtype : Type[int] | Type[float]
        Specifies whether fingerprints will have int or float data type.
    radius : int
        Radius for the Morgan fingerprint.
    n_bits : int
        Number of bits for the Morgan fingerprint.

    Returns
    ----------
    pd.DataFrame
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
