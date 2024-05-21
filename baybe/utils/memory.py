"""Utilities for memory usage."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from baybe.parameters.base import DiscreteParameter
from baybe.utils.numerical import DTypeFloatNumpy


def bytes_to_human_readable(num: float, /) -> tuple[float, str]:
    """Turn a float number representing a memory byte size into a human-readable format.

    Args:
        num: The number representing a memory size in bytes.

    Returns:
        A tuple with the converted number and its determined human-readable unit.
    """
    for unit in ["bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"]:
        if abs(num) < 1024.0:
            return num, unit
        num /= 1024.0
    return num, "YB"


def estimate_discrete_subspace_size(parameters: Iterable[DiscreteParameter]) -> dict:
    """Estimate an upper bound for the search space memory size (ignoring constraints).

    Args:
        parameters: The parameters spanning the search space.

    Returns:
        A dictionary with the searchspace estimation results and units:
            - `Comp_Rep_Size`: Size of the computational representation.
            - `Comp_Rep_Unit`: The unit of Comp_Rep_Size.
            - `Comp_Rep_Shape`: Tuple expressing the shape as (n_rows, n_cols).
            - `Exp_Rep_Size`: Size of the experimental representation.
            - `Exp_Rep_Unit`: The unit of Exp_Rep_Size.
            - `Exp_Rep_Shape`: Tuple expressing the shape as (n_rows, n_cols).
    """
    # Comp rep space is estimated as the size of float times the number of matrix
    # elements in the comp rep. The latter is the total number of value combinations
    # times the total number of columns.
    n_combinations = 1
    n_comp_columns = 0
    for param in parameters:
        n_combinations *= param.comp_df.shape[0]
        n_comp_columns += param.comp_df.shape[1]

    comp_rep_size, comp_rep_unit = bytes_to_human_readable(
        np.array([0.0], dtype=DTypeFloatNumpy).itemsize
        * n_combinations
        * n_comp_columns
    )

    # Exp rep space is estimated as the size of the exp rep dataframe times the number
    # of times it will appear in the entire search space. The latter is the total number
    # of value combination divided by the number of values for the respective parameter.
    # Contributions of all parameters are summed up.
    exp_rep_bytes = 0
    for param in parameters:
        exp_rep_bytes += (
            pd.DataFrame(param.values).memory_usage(index=False, deep=True).sum()
            * n_combinations
            / param.comp_df.shape[0]
        )

    exp_rep_size, exp_rep_unit = bytes_to_human_readable(exp_rep_bytes)

    return {
        "Comp_Rep_Size": comp_rep_size,
        "Comp_Rep_Unit": comp_rep_unit,
        "Comp_Rep_Shape": (n_combinations, n_comp_columns),
        "Exp_Rep_Size": exp_rep_size,
        "Exp_Rep_Unit": exp_rep_unit,
        "Exp_Rep_Shape": (n_combinations, len(parameters)),
    }
