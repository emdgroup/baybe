"""Hypothesis strategies for constraints."""

from functools import partial
from typing import Any

import hypothesis.strategies as st
from hypothesis import assume

from baybe.constraints.conditions import (
    SubSelectionCondition,
    ThresholdCondition,
    _valid_logic_combiners,
)
from baybe.constraints.continuous import (
    ContinuousLinearConstraint,
)
from baybe.constraints.discrete import (
    DiscreteDependenciesConstraint,
    DiscreteExcludeConstraint,
    DiscreteLinkedParametersConstraint,
    DiscreteNoLabelDuplicatesConstraint,
    DiscretePermutationInvarianceConstraint,
    DiscreteProductConstraint,
    DiscreteSumConstraint,
)
from baybe.parameters.base import DiscreteParameter
from baybe.parameters.numerical import NumericalDiscreteParameter

from ..hypothesis_strategies.basic import finite_floats


def sub_selection_conditions(superset: list[Any] | None = None):
    """Generate :class:`baybe.constraints.conditions.SubSelectionCondition`."""
    if superset is None:
        element_strategy = st.text()
    else:
        element_strategy = st.sampled_from(superset)
    return st.builds(
        SubSelectionCondition, st.lists(element_strategy, unique=True, min_size=1)
    )


def threshold_conditions():
    """Generate :class:`baybe.constraints.conditions.ThresholdCondition`."""
    return st.builds(ThresholdCondition, threshold=finite_floats())


@st.composite
def discrete_excludes_constraints(
    draw: st.DrawFn, parameters: list[DiscreteParameter] | None = None
):
    """Generate :class:`baybe.constraints.discrete.DiscreteExcludeConstraint`."""
    if parameters is None:
        parameter_names = draw(st.lists(st.text(), unique=True, min_size=1))
        conditions = draw(
            st.lists(
                st.one_of(sub_selection_conditions(), threshold_conditions()),
                min_size=len(parameter_names),
                max_size=len(parameter_names),
            )
        )

    else:
        parameter_names = [p.name for p in parameters]

        # Threshold conditions only make sense for numerical parameters
        conditions = [
            draw(st.one_of(sub_selection_conditions(p.values), threshold_conditions()))
            if isinstance(p, NumericalDiscreteParameter)
            else draw(sub_selection_conditions(p.values))
            for p in parameters
        ]

    combiner = draw(st.sampled_from(list(_valid_logic_combiners)))
    return DiscreteExcludeConstraint(parameter_names, conditions, combiner)


@st.composite
def discrete_dependencies_constraints(
    draw: st.DrawFn,
    parameters: list[DiscreteParameter] | None = None,
    affected_parameter_names: list[list[str]] | None = None,
):
    if parameters is None:
        # Draw random unique parameter names
        # If affected parameters are given the list length must be respected
        parameter_names = draw(
            st.lists(
                st.text(),
                unique=True,
                min_size=1
                if affected_parameter_names is None
                else len(affected_parameter_names),
                max_size=None
                if affected_parameter_names is None
                else len(affected_parameter_names),
            )
        )
        if affected_parameter_names is not None:
            # Avoid generating parameters that depend on themselves
            assume(
                all(
                    p not in affected_parameter_names[k]
                    for k, p in enumerate(parameter_names)
                )
            )

        conditions = draw(
            st.lists(
                st.one_of(sub_selection_conditions(), threshold_conditions()),
                min_size=len(parameter_names),
                max_size=len(parameter_names),
            )
        )
    else:
        parameter_names = [p.name for p in parameters]

        # Threshold conditions only make sense for numerical parameters
        conditions = [
            draw(st.one_of(sub_selection_conditions(p.values), threshold_conditions()))
            if isinstance(p, NumericalDiscreteParameter)
            else draw(sub_selection_conditions(p.values))
            for p in parameters
        ]

    if affected_parameter_names is None:
        # Draw random lists of dependent parameters, avoiding duplicates with the main
        # parameters
        affected_parameter_names = draw(
            st.lists(
                st.lists(
                    st.text().filter(lambda x: x not in parameter_names),
                    min_size=1,
                ),
                min_size=len(parameter_names),
                max_size=len(parameter_names),
            )
        )
    else:
        # Affected and dependent parameters cannot overlap
        assert all(
            p not in affected_parameter_names[k] for k, p in enumerate(parameter_names)
        ), "Affected parameters cannot overlap with the parameters they depend on"

    return DiscreteDependenciesConstraint(
        parameter_names, conditions, affected_parameter_names
    )


@st.composite
def discrete_permutation_invariance_constraints(
    draw: st.DrawFn,
    parameters: list[DiscreteParameter] | None = None,
    dependencies: DiscreteDependenciesConstraint | None = None,
):
    if parameters is None:
        # Draw random unique parameter names
        parameter_names = draw(st.lists(st.text(), unique=True, min_size=1))
    else:
        parameter_names = [p.name for p in parameters]

    if dependencies is None:
        dependencies = draw(
            st.one_of(
                [
                    st.none(),
                    discrete_dependencies_constraints(
                        parameters=None,
                        affected_parameter_names=[[p] for p in parameter_names],
                    ),
                ]
            )
        )

    return DiscretePermutationInvarianceConstraint(parameter_names, dependencies)


def _discrete_constraints(
    constraint_type: (
        type[DiscreteSumConstraint]
        | type[DiscreteProductConstraint]
        | type[DiscreteNoLabelDuplicatesConstraint]
        | type[DiscreteLinkedParametersConstraint]
    ),
    parameter_names: list[str] | None = None,
):
    """Generate discrete constraints."""
    if parameter_names is None:
        parameters = st.lists(st.text(), unique=True, min_size=1)
    else:
        assert len(parameter_names) > 0
        assert len(parameter_names) == len(set(parameter_names))
        parameters = st.just(parameter_names)

    if constraint_type in [DiscreteSumConstraint, DiscreteProductConstraint]:
        return st.builds(constraint_type, parameters, threshold_conditions())
    else:
        return st.builds(constraint_type, parameters)


discrete_sum_constraints = partial(_discrete_constraints, DiscreteSumConstraint)
"""Generate :class:`baybe.constraints.discrete.DiscreteSumConstraint`."""

discrete_product_constraints = partial(_discrete_constraints, DiscreteProductConstraint)
"""Generate :class:`baybe.constraints.discrete.DiscreteProductConstraint`."""

discrete_no_label_duplicates_constraints = partial(
    _discrete_constraints, DiscreteNoLabelDuplicatesConstraint
)
"""Generate :class:`baybe.constraints.discrete.DiscreteNoLabelDuplicatesConstraint`."""

discrete_linked_parameters_constraints = partial(
    _discrete_constraints, DiscreteLinkedParametersConstraint
)
"""Generate :class:`baybe.constraints.discrete.DiscreteLinkedParametersConstraint`."""


@st.composite
def continuous_linear_constraints(
    draw: st.DrawFn,
    operators: list[str] | None = None,
    parameter_names: list[str] | None = None,
):
    """Generate continuous linear constraints."""  # noqa:E501
    if parameter_names is None:
        parameter_names = draw(st.lists(st.text(), unique=True, min_size=1))
    else:
        assert len(parameter_names) > 0
        assert len(parameter_names) == len(set(parameter_names))

    coefficients = draw(
        st.lists(
            finite_floats(),
            min_size=len(parameter_names),
            max_size=len(parameter_names),
        )
    )
    rhs = draw(finite_floats())

    # Optionally add the operator
    operators = operators or ["=", ">=", "<="]
    operator = draw(st.sampled_from(operators))

    return ContinuousLinearConstraint(parameter_names, operator, coefficients, rhs)


continuous_linear_equality_constraints = partial(
    continuous_linear_constraints, operators=["="]
)
"""Generate linear equality constraints."""

continuous_linear_inequality_constraints = partial(
    continuous_linear_constraints, operators=[">=", "<="]
)
"""Generate linear inequality constraints."""
