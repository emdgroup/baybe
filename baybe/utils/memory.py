"""Utilities for memory usage."""
from typing import cast

from baybe.parameters.base import DiscreteParameter, Parameter
from baybe.utils.numerical import DTypeFloatNumpy


def bytes_to_human_readable(num: float) -> tuple[str, str]:
    """Turn float number representing memory byte size into a human-readable format.

    Args:
        num: The number representing a memory size in bytes.

    Returns:
        Tuple with the converted number string and its determined human-readable unit.
    """
    for unit in ["bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}", unit
        num /= 1024.0
    return f"{num:.1f}", "YB"


def estimate_searchspace_size(parameters: list[Parameter]) -> dict:
    """Estimate upper bound for the search space size in memory.

    For now, constraints are not considered. Since the size of

    Args:
        parameters: List of parameters.

    Returns:
        Dictionary with the results for exp_rep and comp_rep and unit indicator.
    """
    values = 1
    columns = 0
    for param in [p for p in parameters if p.is_discrete]:
        param = cast(DiscreteParameter, param)
        values *= param.comp_df.shape[0]
        columns += param.comp_df.shape[1]

    exp_rep_bytes = 0
    for k, param in enumerate([p for p in parameters if p.is_discrete]):
        param = cast(DiscreteParameter, param)
        exp_rep_bytes += (
            param.comp_df.memory_usage(index=True if k == 0 else False, deep=True).sum()
            * values
            / param.comp_df.shape[0]
        )

    comp_rep_size, comp_rep_unit = bytes_to_human_readable(
        DTypeFloatNumpy(0).itemsize * values * columns
    )

    exp_rep_size, exp_rep_unit = bytes_to_human_readable(exp_rep_bytes)

    return {
        "Comp_Rep_Size": comp_rep_size,
        "Comp_Rep_Unit": comp_rep_unit,
        "Exp_Rep_Size": exp_rep_size,
        "Exp_Rep_Unit": exp_rep_unit,
    }
