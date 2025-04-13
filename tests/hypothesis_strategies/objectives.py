"""Hypothesis strategies for objectives."""

import hypothesis.strategies as st

from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.enum import Scalarizer
from baybe.objectives.pareto import ParetoObjective
from baybe.objectives.single import SingleTargetObjective
from baybe.targets import NumericalTarget, TargetMode
from baybe.targets.numerical import _VALID_TRANSFORMATIONS

from ..hypothesis_strategies.basic import finite_floats
from ..hypothesis_strategies.targets import numerical_targets
from ..hypothesis_strategies.utils import intervals as st_intervals

_intervals = st_intervals(exclude_fully_unbounded=True, exclude_half_bounded=True)

_targets = st.lists(
    numerical_targets(_intervals), min_size=2, unique_by=lambda t: t.name
)


def single_target_objectives():
    """Generate :class:`baybe.objectives.single.SingleTargetObjective`."""
    return st.builds(SingleTargetObjective, target=numerical_targets())


@st.composite
def desirability_objectives(draw: st.DrawFn):
    """Generate :class:`baybe.objectives.desirability.DesirabilityObjective`."""
    targets = draw(_targets)
    weights = draw(
        st.lists(
            finite_floats(min_value=0.0, exclude_min=True),
            min_size=len(targets),
            max_size=len(targets),
        )
    )
    scalarizer = draw(st.sampled_from(Scalarizer))
    return DesirabilityObjective(targets, weights, scalarizer)


@st.composite
def _pareto_target(draw: st.DrawFn):
    """Generate :class:`baybe.targets.numerical.NumericalTarget`.

    Args:
        draw: Hypothesis draw object.

    Returns:
        _type_: _description_
    """
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

    return NumericalTarget(
        name=name, mode=mode, bounds=bounds, transformation=transformation
    )


_pareto_targets = st.lists(_pareto_target(), min_size=2, unique_by=lambda t: t.name)


@st.composite
def pareto_objectives(draw: st.DrawFn):
    """Generate :class:`baybe.objectives.pareto.ParetoObjective`."""
    return ParetoObjective(draw(_pareto_targets))
