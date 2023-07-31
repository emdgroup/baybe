# pylint: disable=missing-function-docstring

"""
Collection of small utilities.
"""
from __future__ import annotations

import logging
import random
import ssl
import urllib.request
from abc import ABC
from dataclasses import dataclass
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
    TypeVar,
    Union,
)

import cattrs
import numpy as np
import pandas as pd
import torch
from attrs import cmp_using
from joblib import Memory
from mordred import Calculator, descriptors
from rdkit import Chem, RDLogger
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from torch import Tensor

if TYPE_CHECKING:
    from baybe.core import BayBE
    from baybe.parameters import Parameter

log = logging.getLogger(__name__)

T = TypeVar("T")

# Data types
# TODO: unclear why pylint wants PascalCase here
DTYPE_FLOAT_NUMPY = np.float64  # pylint: disable=invalid-name
DTYPE_FLOAT_TORCH = torch.float64  # pylint: disable=invalid-name

# Caching related objects
cachedir = Path.home() / ".baybe_cache"
memory_utils = Memory(cachedir / "utils")

# Global Mordred calculator (variable could be replaced with a singleton pattern)
mordred_calculator = Calculator(descriptors)


def isabstract(cls: Any) -> bool:
    """
    Determines if a given class is abstract in a more general sense than
    `inspect.abstract`, which only verifies if a class has abstract methods. The
    latter can be problematic when the class has no abstract methods but is
    nevertheless not directly usable, for example, because it has uninitialized
    members, which are only covered in its non-"abstract" subclasses. By contrast,
    this method simply checks if the class derives from `abc.ABC`.

    Parameters
    ----------
    cls : Any
        The class to be inspected.

    Returns
    -------
    bool
        True if the class is "abstract" (see definition above), False else.
    """
    return ABC in cls.__bases__


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


def to_tensor(*dfs: pd.DataFrame) -> Union[Tensor, Iterable[Tensor]]:
    """Converts a given set of dataframes into tensors (dropping all indices)."""
    # FIXME This function seems to trigger a problem when some columns in either of
    #  the dfs have a dtype other than int or float (e.g. object, bool). This can
    #  weirdly happen, even if all values are numeric, e.g. when a target column is
    #  looked up from a df in simulation, it can have dtype object even if it's all
    #  floats. As a simple fix (this seems to be the most reasonable place to take
    #  care of this) df.values has been changed to df.values.astype(float),
    #  even though this seems like double casting here.
    out = (
        torch.from_numpy(df.values.astype(DTYPE_FLOAT_NUMPY)).to(DTYPE_FLOAT_TORCH)
        for df in dfs
    )
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
                lbound = target.bounds.lower if np.isfinite(target.bounds.lower) else 0
                ubound = (
                    target.bounds.upper if np.isfinite(target.bounds.upper) else 100
                )
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
                lbound = target.bounds.lower if np.isfinite(target.bounds.lower) else 0
                ubound = (
                    target.bounds.upper if np.isfinite(target.bounds.upper) else 100
                )
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
        if param.is_numeric:
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
                    "'absolute' or 'relative_percent'."
                )
            if not param.is_discrete:  # respect continuous intervals
                data[param.name].clip(
                    param.bounds.lower, param.bounds.upper, inplace=True
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


def df_drop_single_value_columns(
    df: pd.DataFrame, lst_exclude: list = None
) -> pd.DataFrame:
    """
    Drops dataframe columns with zero variance.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to be cleaned.
    lst_exclude: list
        List of column names that are excluded from this filter.

    Returns
    -------
    pd.DataFrame
        The cleaned dataframe.
    """
    if lst_exclude is None:
        lst_exclude = []

    to_keep = []
    for col in df.columns:
        if (col in lst_exclude) or (df[col].nunique() > 1):
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


@dataclass(frozen=True, repr=False)
class Dummy:
    """
    Placeholder element for array-like data types. Useful e.g. for detecting
    duplicates in constraints.
    """

    def __repr__(self):
        return "<dummy>"


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


def get_subclasses(cls: T, recursive: bool = True, abstract: bool = False) -> List[T]:
    """
    Returns a list of subclasses for the given class.

    Parameters
    ----------
    cls
        The base class to retrieve subclasses for.
    recursive : bool
        If True, indirect subclasses (i.e. subclasses of subclasses) are included.
    abstract : bool
        If True, abstract subclasses are included.

    Returns
    -------
    list
        A list of subclasses for the given class.
    """
    subclasses = []
    for subclass in cls.__subclasses__():

        # Append direct subclass only if it is not abstract
        if abstract or not isabstract(subclass):
            subclasses.append(subclass)

        # If requested, add indirect subclasses
        if recursive:
            subclasses.extend(get_subclasses(subclass, abstract=abstract))

    return subclasses


def closest_element(array: np.ndarray, target: float) -> float:
    """Finds the element of an array that is closest to a target value."""
    return array[np.abs(array - target).argmin()]


def closer_element(x: float, y: float, target: float) -> float:
    """Determines which of two given inputs is closer to a target value."""
    return x if np.abs(x - target) < np.abs(y - target) else y


def set_random_seed(seed: int) -> None:
    """Sets the global random seed."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def unstructure_base(base):
    converter = cattrs.global_converter
    return {
        "type": base.__class__.__name__,
        **converter.unstructure_attrs_asdict(base),
    }


def get_base_unstructure_hook(base):
    def structure_base(val, _):
        _type = val["type"]
        cls = next((cl for cl in get_subclasses(base) if cl.__name__ == _type), None)
        if cls is None:
            raise ValueError(f"Unknown subclass {_type}.")
        return cattrs.structure_attrs_fromdict(val, cls)

    return structure_base


def eq_dataframe():
    return cmp_using(lambda x, y: x.equals(y))


def fuzzy_row_match(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    parameters: List[Parameter],
    numerical_measurements_must_be_within_tolerance: bool,
) -> pd.Index:
    """
    Matches rows of the right dataframe (e.g. measurements from an experiment)
    to the rows of the left dataframe.

    This is useful for validity checks and to automatically match measurements to
    entries in the search space, e.g. to detect which ones have been measured.
    For categorical parameters, there needs to be an exact match with any of the
    allowed values. For numerical parameters, the user can decide via a flag
    whether values outside the tolerance should be accepted.

    Parameters
    ----------
    left_df : pd.DataFrame
        The data that serves as lookup reference.
    right_df : pd.DataFrame
        The data that should be checked for matching rows in the left data frame.
    parameters : list
        List of baybe parameter objects that are needed to identify potential
        tolerances.
    numerical_measurements_must_be_within_tolerance : bool
        If True, numerical parameters are matched with the search space elements
        only if there is a match within the parameter tolerance. If False,
        the closest match is considered, irrespective of the distance.

    Returns
    -------
    pd.Index
        The index of the matching rows in left_df.
    """

    # Assert that all parameters appear in the given dataframe
    if not all(col in right_df.columns for col in left_df.columns):
        raise ValueError(
            "for fuzzy row matching all rows of the right dataframe need to be present"
            " in the left dataframe."
        )

    inds_matched = []

    # Iterate over all input rows
    for ind, row in right_df.iterrows():
        # Check if the row represents a valid input
        valid = True
        for param in parameters:
            if param.is_numeric:
                if numerical_measurements_must_be_within_tolerance:
                    valid &= param.is_in_range(row[param.name])
            else:
                valid &= param.is_in_range(row[param.name])
            if not valid:
                raise ValueError(
                    f"Input data on row with the index {row.name} has invalid "
                    f"values in parameter '{param.name}'. "
                    f"For categorical parameters, values need to exactly match a "
                    f"valid choice defined in your config. "
                    f"For numerical parameters, a match is accepted only if "
                    f"the input value is within the specified tolerance/range. Set "
                    f"the flag 'numerical_measurements_must_be_within_tolerance' "
                    f"to 'False' to disable this behavior."
                )

        # Differentiate category-like and discrete numerical parameters
        cat_cols = [p.name for p in parameters if not p.is_numeric]
        num_cols = [p.name for p in parameters if (p.is_numeric and p.is_discrete)]

        # Discrete parameters must match exactly
        match = left_df[cat_cols].eq(row[cat_cols]).all(axis=1, skipna=False)

        # For numeric parameters, match the entry with the smallest deviation
        # TODO: allow alternative distance metrics
        for col in num_cols:
            abs_diff = (left_df[col] - row[col]).abs()
            match &= abs_diff == abs_diff.min()

        # We expect exactly one match. If that's not the case, print a warning.
        inds_found = left_df.index[match].to_list()
        if len(inds_found) == 0 and len(num_cols) > 0:
            log.warning(
                "Input row with index %s could not be matched to the search space. "
                "This could indicate that something went wrong.",
                ind,
            )
        elif len(inds_found) > 1:
            log.warning(
                "Input row with index %s has multiple matches with "
                "the search space. This could indicate that something went wrong. "
                "Matching only first occurrence.",
                ind,
            )
            inds_matched.append(inds_found[0])
        else:
            inds_matched.extend(inds_found)

    return pd.Index(inds_matched)


def strtobool(val: str) -> bool:
    """
    Convert a string representation of truth to True or False. Adapted from distutils.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.

    Parameters
    ----------
    val: str
        String to be checked.

    Returns
    -------
    bool
    """
    if val.lower() in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val.lower() in ("n", "no", "f", "false", "off", "0"):
        return False

    raise ValueError(f"Invalid truth value: {val}")
