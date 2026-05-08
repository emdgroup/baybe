"""Utilities for search space construction."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, TypeVar

import pandas as pd

from baybe.constraints import DISCRETE_CONSTRAINTS_FILTERING_ORDER
from baybe.constraints.base import DiscreteConstraint
from baybe.parameters.base import DiscreteParameter

if TYPE_CHECKING:
    import polars as pl

_T = TypeVar("_T")


def select_via_flat_index(flat_idx: int, groups: Sequence[Sequence[_T]]) -> list[_T]:
    """Select one element per group using a flat Cartesian-product index.

    Maps a single integer index over the Cartesian product of ``groups`` to the
    corresponding element from each group, using repeated ``divmod`` to unpack
    the mixed-radix representation.

    Note:
        Given groups of sizes ``[3, 2, 4]`` and ``flat_idx=11``,
        ``divmod(11, 3)`` yields index ``2`` from group 0,
        ``divmod(3, 2)`` yields index ``1`` from group 1, and
        ``divmod(1, 4)`` yields index ``1`` from group 2.

    Args:
        flat_idx: The flat index into the Cartesian product of all groups.
        groups: The groups to select from, one element selected per group.

    Returns:
        A list of selected elements, one per group.
    """
    selected = []
    remaining = flat_idx
    for group in groups:
        remaining, idx = divmod(remaining, len(group))
        selected.append(group[idx])
    return selected


def optimize_parameter_order(
    parameters: Sequence[DiscreteParameter],
    constraints: Sequence[DiscreteConstraint],
) -> list[DiscreteParameter]:
    """Determine a heuristic parameter ordering for incremental space construction.

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
    unconstrained = [p for p in parameters if p.name not in all_constrained_names]

    # Greedy ordering: at each step, pick the parameter whose addition
    # completes (is the last missing parameter for) the most constraints.
    # Ties are broken by picking the parameter with fewest active values
    # (smallest expansion factor during cross-merging).
    ordered: list[DiscreteParameter] = []
    available: set[str] = set()
    remaining = [p for p in parameters if p.name in all_constrained_names]

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
    initial_ldf: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Create the Cartesian product of discrete parameter values using Polars.

    Args:
        parameters: List of discrete parameter objects.
        initial_ldf: An optional starting lazy dataframe. When provided, the
            given parameters are cross-joined into it.

    Returns:
        A lazy dataframe containing all possible discrete parameter value combinations.
    """
    from baybe._optional.polars import polars as pl

    # Convert each parameter to a lazy dataframe for cross-join operation
    param_frames = [pl.LazyFrame({p.name: p.active_values}) for p in parameters]

    # Determine the starting frame
    if initial_ldf is not None:
        res = initial_ldf
    elif param_frames:
        res = param_frames.pop(0)
    else:
        return pl.LazyFrame()

    # Cross-join remaining parameter frames
    for frame in param_frames:
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
    ordered_params = optimize_parameter_order(parameters, filtering_constraints)

    # Determine which parameter names each constraint needs for completion
    pending: list[tuple[DiscreteConstraint, set[str]]] = [
        (c, c._required_parameters) for c in filtering_constraints
    ]

    # Initialize the dataframe
    df = pd.DataFrame() if initial_df is None else initial_df

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


def _apply_constraint_filter_pandas(
    df: pd.DataFrame, constraints: Collection[DiscreteConstraint]
) -> pd.DataFrame:
    """Remove discrete search space entries based on constraints.

    The filtering is done inplace, but the modified object is still returned.

    Args:
        df: The data in experimental representation to be modified inplace.
        constraints: List of discrete constraints.

    Returns:
        The filtered dataframe.
    """
    # Remove entries that violate parameter constraints:
    for constraint in (c for c in constraints if c.eval_during_creation):
        idxs = constraint.get_invalid(df)
        df.drop(index=idxs, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def _apply_constraint_filter_polars(
    ldf: pl.LazyFrame,
    constraints: Sequence[DiscreteConstraint],
) -> pl.LazyFrame:
    """Remove discrete search space entries based on constraints.

    Args:
        ldf: The data in experimental representation to be filtered.
        constraints: Collection of discrete constraints.

    Returns:
        The Polars lazyframe with undesired rows removed.
    """
    for c in constraints:
        to_keep = c.get_invalid_polars().not_()
        ldf = ldf.filter(to_keep)

    return ldf


def build_constrained_product(
    parameters: Sequence[DiscreteParameter],
    constraints: Sequence[DiscreteConstraint],
    initial_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a constrained Cartesian product, using Polars if configured.

    Partitions constraints by Polars support and builds the product accordingly.
    Parameters covered by Polars-capable constraints are cross-joined and
    filtered in Polars first, then the remaining parameters and constraints are
    handled via incremental pandas filtering.

    Args:
        parameters: The discrete parameters to combine.
        constraints: The discrete constraints to apply during construction.
        initial_df: An optional starting dataframe whose columns count as
            already available for constraint evaluation.

    Returns:
        A dataframe containing all valid parameter combinations.
    """
    from baybe.settings import active_settings

    constraints = sorted(
        constraints,
        key=lambda x: DISCRETE_CONSTRAINTS_FILTERING_ORDER.index(x.__class__),
    )

    remaining_params = list(parameters)
    remaining_constraints = list(constraints)

    if active_settings.use_polars_for_constraints:
        from baybe._optional.polars import polars as pl

        polars_constraints = [c for c in constraints if c.has_polars_implementation]

        # Determine which parameters are needed by Polars-capable constraints
        polars_param_names: set[str] = set()
        for c in polars_constraints:
            polars_param_names.update(c._required_parameters)
        polars_params = [p for p in parameters if p.name in polars_param_names]

        if polars_params:
            initial_ldf = (
                pl.from_pandas(initial_df).lazy() if initial_df is not None else None
            )
            lazy_df = parameter_cartesian_prod_polars(
                polars_params, initial_ldf=initial_ldf
            )
            lazy_df = _apply_constraint_filter_polars(lazy_df, polars_constraints)
            initial_df = lazy_df.collect().to_pandas()

            remaining_params = [
                p for p in parameters if p.name not in polars_param_names
            ]
            remaining_constraints = [
                c for c in constraints if not c.has_polars_implementation
            ]

    return parameter_cartesian_prod_pandas_constrained(
        remaining_params, remaining_constraints, initial_df=initial_df
    )
