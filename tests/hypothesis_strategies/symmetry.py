"""Hypothesis strategies for symmetries."""

from itertools import combinations

import hypothesis.strategies as st
from hypothesis import assume

from baybe.parameters.base import Parameter
from baybe.symmetry import DependencySymmetry, MirrorSymmetry, PermutationSymmetry
from tests.hypothesis_strategies.basic import finite_floats
from tests.hypothesis_strategies.conditions import (
    sub_selection_conditions,
    threshold_conditions,
)


@st.composite
def mirror_symmetries(draw: st.DrawFn, parameter_pool: list[Parameter] | None = None):
    """Generate :class:`baybe.symmetry.MirrorSymmetry`."""
    if parameter_pool is None:
        parameter_name = draw(st.text(min_size=1))
    else:
        parameter = draw(st.sampled_from(parameter_pool))
        assume(parameter.is_numerical)
        parameter_name = parameter.name

    return MirrorSymmetry(
        parameter_name=parameter_name,
        use_data_augmentation=draw(st.booleans()),
        mirror_point=draw(finite_floats()),
    )


@st.composite
def permutation_symmetries(
    draw: st.DrawFn, parameter_pool: list[Parameter] | None = None
):
    """Generate :class:`baybe.symmetry.PermutationSymmetry`."""
    if parameter_pool is None:
        parameter_names_pool = draw(
            st.lists(st.text(min_size=1), unique=True, min_size=2)
        )
    else:
        parameter_names_pool = [p.name for p in parameter_pool]

    parameter_names = draw(
        st.lists(st.sampled_from(parameter_names_pool), min_size=2, unique=True).map(
            tuple
        )
    )
    n_params = len(parameter_names)
    n_copermuted_groups = draw(
        st.integers(
            min_value=0,
            max_value=len(parameter_names_pool) // len(parameter_names_pool),
        )
    )
    copermuted_groups = draw(
        st.lists(
            st.lists(
                st.sampled_from(parameter_names_pool),
                unique=True,
                min_size=n_params,
                max_size=n_params,
            ).map(tuple),
            min_size=n_copermuted_groups,
            max_size=n_copermuted_groups,
        ).map(tuple)
    )

    # Ensure no overlap between permutation groups
    for a, b in combinations([parameter_names, *copermuted_groups], 2):
        assume(not set(a) & set(b))

    return PermutationSymmetry(
        parameter_names=parameter_names,
        copermuted_groups=copermuted_groups,
        use_data_augmentation=draw(st.booleans()),
    )


@st.composite
def dependency_symmetries(
    draw: st.DrawFn, parameter_pool: list[Parameter] | None = None
):
    """Generate :class:`baybe.symmetry.DependencySymmetry`."""
    if parameter_pool is None:
        parameter_name = draw(st.text(min_size=1))
        affected_strat = st.lists(
            st.text(min_size=1).filter(lambda x: x != parameter_name),
            min_size=1,
            unique=True,
        ).map(tuple)
    else:
        parameter = draw(st.sampled_from(parameter_pool))
        assume(parameter.is_discrete)
        parameter_name = parameter.name
        affected_strat = st.lists(
            st.sampled_from(parameter_pool)
            .filter(lambda x: x.name != parameter_name)
            .map(lambda x: x.name),
            unique=True,
            min_size=1,
            max_size=len(parameter_pool) - 1,
        ).map(tuple)

    return DependencySymmetry(
        parameter_name=parameter_name,
        condition=draw(st.one_of(threshold_conditions(), sub_selection_conditions())),
        affected_parameter_names=draw(affected_strat),
        n_discretization_points=draw(st.integers(min_value=2)),
        use_data_augmentation=draw(st.booleans()),
    )


symmetries = st.one_of(
    [
        mirror_symmetries(),
        permutation_symmetries(),
        dependency_symmetries(),
    ]
)
"""A strategy that generates symmetries."""
