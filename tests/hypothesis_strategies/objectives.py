"""Hypothesis strategies for objectives."""

from collections.abc import Sequence

import hypothesis.strategies as st

from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.enum import Scalarizer
from baybe.objectives.pareto import ParetoObjective
from baybe.objectives.single import SingleTargetObjective
from baybe.objectives.tfpr import TFPRObjective
from baybe.targets import NumericalTarget
from tests.hypothesis_strategies.basic import finite_floats
from tests.hypothesis_strategies.metadata import metadata
from tests.hypothesis_strategies.targets import numerical_targets
from tests.hypothesis_strategies.targets import target_names as target_name_strategy

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


@st.composite
def tfpr_objectives(draw: st.DrawFn, target_names: Sequence[str] | None = None):
    """Generate :class:`baybe.objectives.tfpr.TFPRObjective`.

    Args:
        draw: Hypothesis draw object.
        target_names: Optional names of the targets to be used. If ``None``, the
            target names are drawn.

    Returns:
        The drawn objective.
    """
    if target_names is None:
        target_names = draw(st.lists(target_name_strategy, min_size=2, unique=True))
    targets = [
        NumericalTarget(name, minimize=draw(st.booleans())) for name in target_names
    ]
    weights = draw(
        st.lists(
            st.integers(min_value=0, max_value=10),
            min_size=len(targets),
            max_size=len(targets),
        ).filter(any)
    )
    tolerances = draw(
        st.lists(
            finite_floats(min_value=0.0, max_value=1.0),
            min_size=len(targets),
            max_size=len(targets),
        )
    )
    optimism_lambda = draw(finite_floats(min_value=0.0, max_value=10.0))
    top_fraction = draw(
        st.none() | finite_floats(min_value=0.0, max_value=1.0, exclude_min=True)
    )
    objective_metadata = draw(metadata())
    return TFPRObjective(
        targets,
        weights,
        tolerances,
        optimism_lambda=optimism_lambda,
        top_fraction=top_fraction,
        metadata=objective_metadata,
    )


objectives = st.one_of(
    single_target_objectives(),
    desirability_objectives(),
    pareto_objectives(),
    tfpr_objectives(),
)
"""A strategy that generates objectives."""
