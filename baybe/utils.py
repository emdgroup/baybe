"""
Collection of small utilities
"""

import logging
from typing import Iterable, Union

import pandas as pd
import torch
from torch import Tensor

log = logging.getLogger(__name__)


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
