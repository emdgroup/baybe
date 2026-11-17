"""Hypothesis strategies for objectives."""

import hypothesis.strategies as st

from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.enum import Scalarizer
from baybe.objectives.pareto import ParetoObjective
from baybe.objectives.single import SingleTargetObjective
from tests.hypothesis_strategies.basic import finite_floats
from tests.hypothesis_strategies.metadata import metadata
from tests.hypothesis_strategies.targets import numerical_targets

_target_lists = st.lists(numerical_targets(), min_size=2, unique_by=lambda t: t.name)
_normalized_target_lists = st.lists(
    numerical_targets(normalized=True), min_size=2, unique_by=lambda t: t.name
)


@st.composite
def single_target_objectives(draw: st.DrawFn):
    """Generate :class:`baybe.objectives.single.SingleTargetObjective`."""
    target = draw(numerical_targets())
    objective_metadata = draw(metadata())
    return SingleTargetObjective(target=target, metadata=objective_metadata)


@st.composite
def desirability_objectives(draw: st.DrawFn):
    """Generate :class:`baybe.objectives.desirability.DesirabilityObjective`."""
    scalarizer = draw(st.sampled_from(Scalarizer))
    if require_normalization := (
        draw(st.booleans()) or scalarizer is Scalarizer.GEOM_MEAN
    ):
        targets = draw(_normalized_target_lists)
    else:
        targets = draw(_target_lists)
    weights = draw(
        st.lists(
            finite_floats(min_value=0.0, exclude_min=True),
            min_size=len(targets),
            max_size=len(targets),
        )
    )
    objective_metadata = draw(metadata())
    return DesirabilityObjective(
        targets,
        weights,
        scalarizer,
        require_normalization=require_normalization,
        metadata=objective_metadata,
    )


@st.composite
def pareto_objectives(draw: st.DrawFn):
    """Generate :class:`baybe.objectives.pareto.ParetoObjective`."""
    objective_metadata = draw(metadata())
    targets = draw(_target_lists)
    return ParetoObjective(targets, metadata=objective_metadata)


objectives = st.one_of(
    single_target_objectives(),
    desirability_objectives(),
    pareto_objectives(),
)
"""A strategy that generates objectives."""
