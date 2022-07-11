"""
Collection of small utilities
"""

from typing import Iterable, Union

import pandas as pd
import torch
from torch import Tensor


def is_valid_smiles(smiles: str):
    """
    Test if a SMILEs string is valid. Currently NOT IMPLEMENTED
    :param smiles: SMILES string to test
    :return: True if the SMILES is valid according to RDKit
    """
    return smiles


def to_tensor(*dfs: pd.DataFrame) -> Union[Tensor, Iterable[Tensor]]:
    """Converts a given set of dataframes into tensors (dropping all indices)."""
    out = (torch.from_numpy(df.values).to(torch.float32) for df in dfs)
    if len(dfs) == 1:
        out = next(out)
    return out
