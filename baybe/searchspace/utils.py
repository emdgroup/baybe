"""Utilities for search space construction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

import narwhals.stable.v2 as nw
import numpy as np

from baybe.constraints import DISCRETE_CONSTRAINTS_FILTERING_ORDER
from baybe.constraints.base import DiscreteConstraint, DiscreteFilteringConstraint
from baybe.parameters.base import DiscreteParameter
from baybe.utils.dataframe import Filter, _filter_dataframe, _infer_backend

if TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoFrame

_T = TypeVar("_T")


def select_via_flat_index(flat_idx: int, /, groups: Sequence[Sequence[_T]]) -> list[_T]:
    """Select one element per group using a flat Cartesian-product index.

    Maps a single integer index over the Cartesian product of ``groups`` to the
    corresponding element from each group via :func:`numpy.unravel_index`.

    Examples:
        >>> groups = [[0, 1], ["a", "b"]]
        >>> select_via_flat_index(0, groups)
        [0, 'a']
        >>> select_via_flat_index(1, groups)
        [0, 'b']
        >>> select_via_flat_index(2, groups)
        [1, 'a']
        >>> select_via_flat_index(3, groups)
        [1, 'b']

    Args:
        flat_idx: The flat index into the Cartesian product of all groups.
        groups: The groups to select from, one element selected per group.

    Returns:
        A list of selected elements, one per group.
    """
    shape = tuple(len(g) for g in groups)
    indices = np.unravel_index(flat_idx, shape)
    return [group[int(idx)] for group, idx in zip(groups, indices)]


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


def _apply_evaluable_constraints(
    frame: nw.LazyFrame,
    constraints: Sequence[DiscreteFilteringConstraint],
) -> tuple[nw.LazyFrame, list[DiscreteFilteringConstraint]]:
    """Apply all evaluable constraints and return those still pending.

    Args:
        frame: The frame to filter.
        constraints: The constraints to consider.

    Returns:
        The filtered frame and the constraints whose required parameters are not yet
        all available.
    """
    available = set(frame.collect_schema().names())
    pending: list[DiscreteFilteringConstraint] = []
    for constraint in constraints:
        if constraint._can_evaluate(available):
            frame = constraint._filter(frame)
        if constraint._required_parameters - available:
            pending.append(constraint)
    return frame, pending


def _sort_filters(filters: Sequence[Filter]) -> list[Filter]:
    """Order filtering constraints canonically, moving all other filters last."""
    constraints = sorted(
        (f for f in filters if isinstance(f, DiscreteFilteringConstraint)),
        key=lambda c: DISCRETE_CONSTRAINTS_FILTERING_ORDER.index(c.__class__),
    )
    others = [f for f in filters if not isinstance(f, DiscreteFilteringConstraint)]
    return [*constraints, *others]


def _make_filtered_cartesian_product(
    parameters: Sequence[DiscreteParameter],
    filters: Sequence[Filter],
    initial_df: IntoFrame | None = None,
    *,
    sort_filters: bool = False,
) -> nw.LazyFrame:
    """Create the Cartesian parameter product and apply the given filters.

    Args:
        parameters: The discrete parameters to combine.
        filters: The filters to apply to the product, in the given order.
        initial_df: An optional starting frame, whose columns are cross-joined with the
            parameter values.
        sort_filters: If ``True``, apply filters in their canonical order
            (see :data:`DISCRETE_CONSTRAINTS_FILTERING_ORDER`) instead of the
            order in which they are passed.

    Returns:
        The filtered Narwhals lazy frame.
    """
    if sort_filters:
        filters = _sort_filters(filters)

    backend = _infer_backend(initial_df)

    frames = [
        nw.new_series(parameter.name, parameter.active_values, backend=backend)
        .to_frame()
        .lazy()
        for parameter in parameters
    ]
    if initial_df is not None:
        frames.insert(0, nw.from_native(initial_df).lazy())
    if not frames:
        return nw.from_dict({}, backend=backend).lazy()

    product = frames[0]
    for frame in frames[1:]:
        product = product.join(frame, how="cross")

    return _filter_dataframe(product, *filters)


def _make_filtered_cartesian_product_incremental(
    parameters: Sequence[DiscreteParameter],
    filters: Sequence[Filter],
    initial_df: IntoFrame | None = None,
    *,
    sort_filters: bool = False,
    sort_parameters: bool = False,
) -> nw.LazyFrame:
    """Create a Cartesian parameter product, applying filters as early as possible.

    Unlike :func:`_make_filtered_cartesian_product`, which builds the full product
    before filtering, this variant cross-joins parameters one at a time and applies
    each filtering constraint as soon as it can be (partially) evaluated. Constraints
    are re-evaluated after every join until all of their required parameters are
    available.

    Args:
        parameters: The discrete parameters to combine.
        filters: The filters to apply to the product. Filtering constraints are
            applied incrementally; all other filters are applied at the end.
        initial_df: An optional starting frame, whose columns are cross-joined with
            the parameter values.
        sort_filters: If ``True``, apply filters in their canonical order
            (see :data:`DISCRETE_CONSTRAINTS_FILTERING_ORDER`) instead of the
            order in which they are passed.
        sort_parameters: If ``True``, join parameters in the order given by
            :func:`optimize_parameter_order`. Otherwise, keep the given order.

    Returns:
        The filtered Narwhals lazy frame.
    """
    if sort_filters:
        filters = _sort_filters(filters)

    backend = _infer_backend(initial_df)

    constraints = [f for f in filters if isinstance(f, DiscreteFilteringConstraint)]
    other_filters = [
        f for f in filters if not isinstance(f, DiscreteFilteringConstraint)
    ]
    ordered_parameters = (
        optimize_parameter_order(parameters, constraints)
        if sort_parameters
        else list(parameters)
    )

    frames = [
        nw.new_series(parameter.name, parameter.active_values, backend=backend)
        .to_frame()
        .lazy()
        for parameter in ordered_parameters
    ]
    if initial_df is not None:
        frames.insert(0, nw.from_native(initial_df).lazy())

    if not frames:
        return nw.from_dict({}, backend=backend).lazy()

    product = frames[0]
    product, pending = _apply_evaluable_constraints(product, constraints)

    for column in frames[1:]:
        product = product.join(column, how="cross")
        product, pending = _apply_evaluable_constraints(product, pending)

    # All constraints must have been fully applied after the last join
    assert not pending

    # Restore the original column order (incremental construction may reorder them)
    if parameters:
        initial_columns = (
            nw.from_native(initial_df).columns if initial_df is not None else []
        )
        columns = [*initial_columns, *(parameter.name for parameter in parameters)]
        product = product.select(columns)

    return _filter_dataframe(product, *other_filters)
