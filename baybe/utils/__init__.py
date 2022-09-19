"""
Collection of small utilities.
"""
from __future__ import annotations

import binascii
import pickle

import ssl
import urllib.request
from functools import lru_cache
from pathlib import Path

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

import numpy as np
import pandas as pd
import torch

from joblib import Memory
from mordred import Calculator, descriptors
from rdkit import Chem, RDLogger
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from torch import Tensor

if TYPE_CHECKING:
    from .core import BayBE  # TODO: fix unresolved import

# Caching related objects
cachedir = Path.home() / ".baybe_cache"
memory_utils = Memory(cachedir / "utils")

# Global Mordred calculator (variable could be replaced with a singleton pattern)
mordred_calculator = Calculator(descriptors)


def is_valid_smiles(smiles: str) -> bool:
    """
    Tests if a SMILES string is valid according to RDKit.

    Parameters
    ----------
    smiles : str
        SMILES string to be tested.

    Returns
    -------
    bool
        True if smiles is valid, False else.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def to_tensor(*dfs: Iterable[pd.DataFrame]) -> Union[Tensor, Iterable[Tensor]]:
    """Converts a given set of dataframes into tensors (dropping all indices)."""
    # FIXME This function seems to trigger a problem when some columns in either of
    #  the dfs have a dtype other than int or float (e.g. object, bool). This can
    #  weirdly happen, even if all values are numeric, e.g. when a target column is
    #  looked up from a df in simulation, it can have dtype object even if it's all
    #  floats. As a simple fix (this seems to be the most reasonable place to take
    #  care of this) df.values has been changed to df.values.astype(float),
    #  even though this seems like double casting here.
    out = (torch.from_numpy(df.values.astype(float)).to(torch.float32) for df in dfs)
    if len(dfs) == 1:
        out = next(out)
    return out


def check_if_in(element: Any, allowed: list) -> None:
    """
    Checks if an element is in a given list of elements and raises a
    context-specific exception if it is not.
    """
    if element not in allowed:
        raise ValueError(
            f"The value '{element}' is not allowed. Must be one of {allowed}."
        )


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


def add_fake_results(
    data: pd.DataFrame,
    baybe: BayBE,
    good_reference_values: Optional[Dict[str, list]] = None,
    good_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
    bad_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """
    Adds fake results to a dataframe which was the result of the BayBE recommendation
    action. It is possible to specify "good" values, which will be given a better
    target value. With this, the algorithm can be driven towards certain optimal values
    whilst still being random. Useful for testing.

    Parameters
    ----------
    data : pd.DataFrame
        Output of the `recommend` function of a `BayBE` object.
    baybe : BayBE
        The `BayBE` object, which provides configuration, targets, etc.
    good_reference_values : dict (optional)
        A dictionary containing parameter names (= dict keys) and respective
        parameter values (= dict values) that specify what will be considered good
        parameter settings. Conditions for different parameters are connected via
        "and" logic, i.e. the targets will only get good values when all parameters
        have good reference values.
    good_intervals : dict (optional)
        A dictionary containing target names (= dict keys) and respective "good"
        target value ranges (= dict values) in the form of 2-tuples. Each target will
        be assigned a random value in its respective target range whenever the
        corresponding parameters meet the conditions specified through
        `good_reference_values`.
    bad_intervals : dict (optional)
        Analogous to `good_intervals` but covering the cases where the parameters lie
        outside the conditions specified through `good_reference_values`.

    Example
    -------
    good_reference_values = {'Param1': [1, 4, 42], 'Param2': ['A', 'C']}.
    good_intervals = {'Target1': (5, 10), 'Target2': (0, 100)}
    bad_intervals = {'Target1': (0, 5), 'Target2': (-99, 0)}
    data = baybe.recommend(batch_quantity=10)
    add_fake_results(data, baybe, good_reference_values, good_intervals, bad_intervals)

    Returns
    -------
    Nothing (the given dataframe is modified in-place).
    """
    # Per default, there are no reference values for good parameters
    if good_reference_values is None:
        good_reference_values = {}

    # Validate input
    for param, vals in good_reference_values.items():
        if param not in data.columns:
            raise ValueError(
                f"When adding fake results you specified good "
                f"values for the parameter '{param}' but this "
                f"parameter is not in the dataframe."
            )
        if not isinstance(vals, list):
            raise TypeError(
                f"Entries in parameter 'good_reference_values' "
                f"(which is a dictionary) must be lists, but you "
                f"provided {vals}."
            )

    # Set defaults for good intervals
    if good_intervals is None:
        good_intervals = {}
        for target in baybe.targets:
            if target.mode == "MAX":
                interv = (66, 100)
            elif target.mode == "MIN":
                interv = (0, 33)
            elif target.mode == "MATCH":
                lbound = 0 if target.bounds is None else target.bounds[0]
                ubound = 100 if target.bounds is None else target.bounds[1]
                interv = (
                    lbound + 0.33 * (ubound - lbound),
                    lbound + 0.66 * (ubound - lbound),
                )
            else:
                raise ValueError(
                    "Unrecognized target mode when trying to add fake values."
                )
            good_intervals[target.name] = interv

    # Set defaults for bad intervals
    if bad_intervals is None:
        bad_intervals = {}
        for target in baybe.targets:
            if target.mode == "MAX":
                interv = (0, 33)
            elif target.mode == "MIN":
                interv = (66, 100)
            elif target.mode == "MATCH":
                lbound = 0 if target.bounds is None else target.bounds[0]
                ubound = 100 if target.bounds is None else target.bounds[1]
                interv = (
                    # Take as bad values and arbitrary interval above the match interval
                    ubound + 0.5 * (ubound - lbound),
                    ubound + 2.0 * (ubound - lbound),
                )
            else:
                raise ValueError(
                    "Unrecognized target mode when trying to add fake values."
                )
            bad_intervals[target.name] = interv

    # Add the fake data for each target
    for target in baybe.targets:

        # Add bad values
        data[target.name] = np.random.randint(
            bad_intervals[target.name][0], bad_intervals[target.name][1], len(data)
        )

        # Create masks that identify locations where to place good values
        masks = []
        for param, vals in good_reference_values.items():
            mask = data[param].isin(vals)
            masks.append(mask)

        # Overwrite bad values with good ones using the computed masks
        if len(masks) > 0:
            final_mask = pd.concat(masks, axis=1).all(axis=1)
            data.loc[final_mask, target.name] = np.random.randint(
                good_intervals[target.name][0],
                good_intervals[target.name][1],
                final_mask.sum(),
            )


def add_parameter_noise(
    data: pd.DataFrame,
    baybe: BayBE,
    noise_type: Literal["absolute", "relative_percent"] = "absolute",
    noise_level: float = 1.0,
) -> None:
    """
    Applies uniform noise (additive or multiplicative) to the parameter values of a
    recommendation frame. This can be used to simulate experimental noise or
    imperfect user input containing numerical parameter values that differ from the
    recommendations.

    Parameters
    ----------
    data : pd.DataFrame
        Output of the `recommend` function of a `BayBE` object.
    baybe : BayBE
        The `BayBE` object, which provides configuration, targets, etc.
    noise_type : "absolute" | "relative_percent"
        Defines whether the noise should be additive or multiplicative.
    noise_level : float
        Level/magnitude of the noise. Must be provided as numerical value for
        noise type 'absolute' and as percentage for noise type 'relative_percent'.

    Returns
    -------
    Nothing (the given dataframe is modified in-place).
    """
    for param in baybe.parameters:
        if "NUM" in param.type:
            if noise_type == "relative_percent":
                data[param.name] *= np.random.uniform(
                    1.0 - noise_level / 100.0, 1.0 + noise_level / 100.0, len(data)
                )
            elif noise_type == "absolute":
                data[param.name] += np.random.uniform(
                    -noise_level, noise_level, len(data)
                )
            else:
                raise ValueError(
                    f"Parameter 'noise_type' was {noise_type} but must be either "
                    f"'absolute' or 'relative_percent'."
                )


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
    columns = [prefix + str(name) for name in descriptor_names]
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
        desc = {prefix + dname: func(mol) for dname, func in Chem.Descriptors.descList}
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
            {prefix + "FP" + f"{k + 1}": dtype(bit) for k, bit in enumerate(fpvec)}
        )

    df = pd.DataFrame(res)

    return df


def df_drop_single_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops dataframe columns with zero variance.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be cleaned.

    Returns
    -------
    pd.DataFrame
        The cleaned dataframe.
    """
    to_keep = []
    for col in df:
        if len(df[col].drop_duplicates()) > 1:
            to_keep.append(col)

    return df[to_keep]


def df_drop_string_columns(
    df: pd.DataFrame, ignore_list: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Drops dataframe columns with string values.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be cleaned.
    ignore_list : List[str] (optional)
        List of columns that should not be dropped, even if they include string values.

    Returns
    -------
    pd.DataFrame
        The cleaned dataframe.
    """
    ignore_list = ignore_list or []
    no_string = ~df.applymap(lambda x: isinstance(x, str)).any()
    no_string = no_string[no_string].index
    to_keep = set(no_string).union(set(ignore_list))
    ordered_cols = [col for col in df if col in to_keep]
    return df[ordered_cols]


def df_uncorrelated_features(
    df: pd.DataFrame, exclude_list: Optional[List[str]] = None, threshold: float = 0.7
):
    """
    Returns an uncorrelated set of features. Adapted from edbo.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be cleaned
    exclude_list : list of strings or None
        If provided this defines the columns that should be ignored
    threshold : float
        Threshold for column-column correlation above which columns should be dropped
    Returns
    -------
    data : pandas.DataFrame
        A new dataframe
    """
    # TODO: revise or replace with VRE method

    if exclude_list is None:
        data = df.copy()
    else:
        data = df.drop(columns=exclude_list)

    corr = data.corr().abs()
    to_keep = []
    for i in range(len(corr.iloc[:, 0])):
        above = corr.iloc[:i, i]
        if len(to_keep) > 0:
            above = above[to_keep]
        if len(above[above < threshold]) == len(above):
            to_keep.append(corr.columns.values[i])

    data = data[to_keep]

    if exclude_list is not None:
        data[exclude_list] = df.loc[:, exclude_list]

    return data


class StrictValidationError(Exception):
    """
    This class is used as a workaround that can be used to stop pydantic from continuing
    validating other members when an earlier validation failed.

    The issue is described here:
    https://github.com/pydantic/pydantic/issues/3915
    """


def geom_mean(arr: np.ndarray, weights: List[float] = None) -> np.ndarray:
    """
    Calculates the (weighted) geometric mean along the second axis of a given 2-D array.
    Alternative to `gmean` from scipy that avoids logarithms and division errors.

    Parameters
    ----------
    arr : np.ndarray
        The array containing the values for the mean computation.
    weights : List[float] (optional)
        Optional weights for the mean computation.

    Returns
    -------
    np.ndarray
        A 1-D array containing the row-wise geometric means of the given array.
    """
    return np.prod(np.power(arr, np.atleast_2d(weights) / np.sum(weights)), axis=1)


class HashableDict(dict):
    """Allows hashing of (nested) dictionaries."""

    # TODO: maybe there is a smarter way to achieve the same goal?

    def __hash__(self) -> int:
        return int(binascii.hexlify(pickle.dumps(self)), 16)
