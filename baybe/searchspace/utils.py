"""Utilities for search space construction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

from baybe.constraints.base import DiscreteConstraint
from baybe.parameters.base import DiscreteParameter

if TYPE_CHECKING:
    import polars as pl


def compute_parameter_order(
    parameters: Sequence[DiscreteParameter],
    constraints: Sequence[DiscreteConstraint],
) -> list[DiscreteParameter]:
    """Compute an optimal parameter ordering for incremental space construction.

    Parameters involved in constraints are placed first, ordered so that the
    parameters completing the most constraints come earliest. Parameters not
    involved in any constraint are placed last.

    Args:
        parameters: The discrete parameters.
        constraints: The discrete constraints.

    Returns:
        The parameters in an order optimized for incremental constraint
        filtering.
    """
    if not constraints:
        return list(parameters)

    # Compute which parameter names each constraint needs
    constraint_params = [c._required_parameters for c in constraints]

    # Separate constrained from unconstrained parameters
    all_constrained_names = set().union(*constraint_params)
    constrained = [p for p in parameters if p.name in all_constrained_names]
    unconstrained = [p for p in parameters if p.name not in all_constrained_names]

    # Greedy ordering: at each step, pick the parameter whose addition
    # completes (is the last missing parameter for) the most constraints.
    # Ties are broken by picking the parameter with fewest active values
    # (smallest expansion factor during cross-merging).
    ordered: list[DiscreteParameter] = []
    available: set[str] = set()
    remaining = list(constrained)

    while remaining:
        best_param = None
        best_score = (-1, float("inf"))  # (completions, -active_values)

        for param in remaining:
            candidate_available = available | {param.name}
            completions = sum(
                1
                for cp in constraint_params
                if cp <= candidate_available and not cp <= available
            )
            n_values = len(param.active_values)
            score = (completions, -n_values)
            if score > best_score:
                best_score = score
                best_param = param

        assert best_param is not None
        ordered.append(best_param)
        available.add(best_param.name)
        remaining.remove(best_param)

    # Unconstrained parameters go last
    ordered.extend(unconstrained)
    return ordered


def parameter_cartesian_prod_polars(
    parameters: Sequence[DiscreteParameter],
) -> pl.LazyFrame:
    """Create the Cartesian product of discrete parameter values using Polars.

    Args:
        parameters: List of discrete parameter objects.

    Returns:
        A lazy dataframe containing all possible discrete parameter value combinations.
    """
    from baybe._optional.polars import polars as pl

    if not parameters:
        return pl.LazyFrame()

    # Convert each parameter to a lazy dataframe for cross-join operation
    param_frames = [pl.LazyFrame({p.name: p.active_values}) for p in parameters]

    # Handling edge cases
    if len(param_frames) == 1:
        return param_frames[0]

    # Cross-join parameters
    res = param_frames[0]
    for frame in param_frames[1:]:
        res = res.join(frame, how="cross", force_parallel=True)

    return res


def parameter_cartesian_prod_pandas(
    parameters: Sequence[DiscreteParameter],
) -> pd.DataFrame:
    """Create the Cartesian product of discrete parameter values using Pandas.

    Args:
        parameters: List of discrete parameter objects.

    Returns:
        A dataframe containing all possible discrete parameter value combinations.
    """
    if not parameters:
        return pd.DataFrame()

    index = pd.MultiIndex.from_product(
        [p.active_values for p in parameters], names=[p.name for p in parameters]
    )
    ret = pd.DataFrame(index=index).reset_index()

    return ret


def parameter_cartesian_prod_pandas_constrained(
    parameters: Sequence[DiscreteParameter],
    constraints: Sequence[DiscreteConstraint],
    initial_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a Cartesian product of parameters with incremental constraint filtering.

    Instead of creating the full Cartesian product and then filtering, this
    function cross-merges parameters one by one, applying constraint filters
    as early as possible. This significantly reduces memory usage and
    construction time for highly constrained spaces.

    Parameters are ordered so that constrained parameters come first, enabling
    constraints to fire early when the intermediate dataframe is still small.

    Args:
        parameters: The discrete parameters to combine.
        constraints: The discrete constraints to apply.
        initial_df: An optional starting dataframe. When provided, the given
            parameters are cross-merged into it (its columns count as already
            available for constraint evaluation).

    Returns:
        A dataframe containing all valid parameter combinations.
    """
    # Filter to constraints that should be applied during creation
    filtering_constraints = [c for c in constraints if c.eval_during_creation]

    # Fast path: no constraints and no initial dataframe
    if not filtering_constraints and initial_df is None:
        return parameter_cartesian_prod_pandas(parameters)

    # Compute optimal parameter order
    ordered_params = compute_parameter_order(parameters, filtering_constraints)

    # Determine which parameter names each constraint needs for completion
    pending: list[tuple[DiscreteConstraint, set[str]]] = [
        (c, c._required_parameters) for c in filtering_constraints
    ]

    # Initialize the dataframe
    if initial_df is not None:
        df = initial_df
    else:
        df = pd.DataFrame()

    # Original column order for final reindexing
    original_columns = (list(initial_df.columns) if initial_df is not None else []) + [
        p.name for p in parameters
    ]

    # Incremental cross-merge loop
    for param in ordered_params:
        param_df = pd.DataFrame({param.name: param.active_values})
        if df.empty:
            df = param_df
        else:
            df = pd.merge(df, param_df, how="cross")

        available = set(df.columns)
        still_pending: list[tuple[DiscreteConstraint, set[str]]] = []

        for constraint, all_params in pending:
            idxs = constraint.get_invalid(df, allow_missing=True)
            df.drop(index=idxs, inplace=True)

            if not (all_params <= available):
                still_pending.append((constraint, all_params))

        pending = still_pending

    # Apply any remaining constraints whose parameters were already present
    # in the initial_df (i.e., no new parameters were needed to complete them)
    if pending and not df.empty:
        available = set(df.columns)
        for constraint, all_params in pending:
            if all_params <= available:
                idxs = constraint.get_invalid(df)
                df.drop(index=idxs, inplace=True)

    # Reorder columns and reset index
    if original_columns:
        df = df[original_columns]
    df.reset_index(drop=True, inplace=True)

    return df
