"""Utilities for memory usage."""
from typing import cast

import pandas as pd

from baybe.parameters.base import DiscreteParameter, Parameter
from baybe.utils.numerical import DTypeFloatNumpy


def bytes_to_human_readable(num: float) -> tuple[float, str]:
    """Turn float number representing memory byte size into a human-readable format.

    Args:
        num: The number representing a memory size in bytes.

    Returns:
        Tuple with the converted number and its determined human-readable unit.
    """
    for unit in ["bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"]:
        if abs(num) < 1024.0:
            return num, unit
        num /= 1024.0
    return num, "YB"


def estimate_searchspace_size(parameters: list[Parameter]) -> dict:
    """Estimate upper bound for the search space size in memory.

    For now, constraints are not considered.

    Args:
        parameters: List of parameters.

    Returns:
        Dictionary with the results for exp_rep and comp_rep and unit indicator.
    """
    # Comp rep space is estimated as the size of float times the number of matrix
    # elements in the comp rep. The latter is the total number of value combinations
    # times the number of total columns.
    n_combinations = 1
    n_columns = 0
    for param in [p for p in parameters if p.is_discrete]:
        param = cast(DiscreteParameter, param)
        n_combinations *= param.comp_df.shape[0]
        n_columns += param.comp_df.shape[1]

    comp_rep_size, comp_rep_unit = bytes_to_human_readable(
        DTypeFloatNumpy(0).itemsize * n_combinations * n_columns
    )

    # The exp rep is estimated as the size of the exp rep dataframe times the number of
    # times it will appear in the entire search space. The latter is the total number
    # of value combination divided by the number of values for the respective parameter.
    # Contributions of all parameters are summed up.
    exp_rep_bytes = 0
    for k, param in enumerate([p for p in parameters if p.is_discrete]):
        param = cast(DiscreteParameter, param)
        exp_rep_bytes += (
            pd.DataFrame(param.values).memory_usage(index=False, deep=True).sum()
            * n_combinations
            / param.comp_df.shape[0]
        )

    exp_rep_size, exp_rep_unit = bytes_to_human_readable(exp_rep_bytes)

    return {
        "Comp_Rep_Size": comp_rep_size,
        "Comp_Rep_Unit": comp_rep_unit,
        "Exp_Rep_Size": exp_rep_size,
        "Exp_Rep_Unit": exp_rep_unit,
    }
