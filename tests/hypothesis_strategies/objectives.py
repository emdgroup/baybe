"""Hypothesis strategies for objectives."""

import hypothesis.strategies as st

from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.enum import Scalarizer
from baybe.objectives.pareto import ParetoObjective
from baybe.objectives.single import SingleTargetObjective
from baybe.targets import NumericalTarget, TargetMode
from baybe.targets.numerical import _VALID_TRANSFORMATIONS

from ..hypothesis_strategies.basic import finite_floats
from ..hypothesis_strategies.metadata import measurable_metadata, metadata
from ..hypothesis_strategies.targets import numerical_targets
from ..hypothesis_strategies.utils import intervals as st_intervals

_intervals = st_intervals(exclude_fully_unbounded=True, exclude_half_bounded=True)

_target_lists = st.lists(
    numerical_targets(_intervals), min_size=2, unique_by=lambda t: t.name
)


@st.composite
def single_target_objectives(draw: st.DrawFn):
    """Generate :class:`baybe.objectives.single.SingleTargetObjective`."""
    target = draw(numerical_targets())
    objective_metadata = draw(st.one_of(st.none(), metadata()))
    return SingleTargetObjective(target=target, metadata=objective_metadata)


@st.composite
def desirability_objectives(draw: st.DrawFn):
    """Generate :class:`baybe.objectives.desirability.DesirabilityObjective`."""
    targets = draw(_target_lists)
    weights = draw(
        st.lists(
            finite_floats(min_value=0.0, exclude_min=True),
            min_size=len(targets),
            max_size=len(targets),
        )
    )
    scalarizer = draw(st.sampled_from(Scalarizer))
    objective_metadata = draw(st.one_of(st.none(), metadata()))
    return DesirabilityObjective(
        targets, weights, scalarizer, metadata=objective_metadata
    )


@st.composite
def _pareto_targets(draw: st.DrawFn):
    """Generate :class:`baybe.targets.numerical.NumericalTarget`."""
    name = draw(st.text(min_size=1))
    mode = draw(st.sampled_from(TargetMode))

    if mode is TargetMode.MATCH:
        transformation = draw(st.sampled_from(_VALID_TRANSFORMATIONS[mode]))
        bounds = draw(
            st_intervals(exclude_half_bounded=True, exclude_fully_unbounded=True)
        )
    else:
        transformation = None
        bounds = None

    # Optionally generate metadata for Pareto targets
    target_metadata = draw(st.one_of(st.none(), measurable_metadata()))

    return NumericalTarget(
        name=name,
        mode=mode,
        bounds=bounds,
        transformation=transformation,
        metadata=target_metadata,
    )


_pareto_target_lists = st.lists(
    _pareto_targets(), min_size=2, unique_by=lambda t: t.name
)


@st.composite
def pareto_objectives(draw: st.DrawFn):
    """Generate :class:`baybe.objectives.pareto.ParetoObjective`."""
    targets = draw(_pareto_target_lists)
    objective_metadata = draw(st.one_of(st.none(), metadata()))
    return ParetoObjective(targets, metadata=objective_metadata)
