"""
Collection of small utilities
"""

from typing import Any, Iterable, Union

import pandas as pd
import torch
from torch import Tensor


def is_valid_smiles(smiles: str) -> bool:
    """
    Test if a SMILEs string is valid according to RDKit.

    Parameters
    ----------
    smiles: str
        SMILES string
    Returns
    -------
        bool, True if smiles is valid, False else
    """
    raise NotImplementedError("This function is not implemented yet")


def to_tensor(*dfs: pd.DataFrame) -> Union[Tensor, Iterable[Tensor]]:
    """Converts a given set of dataframes into tensors (dropping all indices)."""
    out = (torch.from_numpy(df.values).to(torch.float32) for df in dfs)
    if len(dfs) == 1:
        out = next(out)
    return out


def check_if_in(element: Any, allowed: list):
    """
    Checks if an element is in a given list of elements and raises a
    context-specific exception if it is not.
    """
    if element not in allowed:
        raise ValueError(
            f"The value '{element}' is not allowed. Must be one of {allowed}."
        )
