"""Hypothesis strategies for constraints."""

from functools import partial

import hypothesis.strategies as st
from hypothesis import assume

from baybe.constraints.conditions import (
    _threshold_operators,
    _valid_logic_combiners,
    _valid_tolerance_operators,
)
from baybe.constraints.continuous import (
    ContinuousLinearConstraint,
)
from baybe.constraints.discrete import (
    DiscreteDependenciesConstraint,
    DiscreteLinearConstraint,
    DiscretePermutationInvarianceConstraint,
    DiscreteProductConstraint,
    DiscreteRepetitionConstraint,
    DiscreteSelectionConstraint,
)
from baybe.parameters.base import DiscreteParameter
from baybe.parameters.numerical import NumericalDiscreteParameter
from tests.hypothesis_strategies.basic import finite_floats
from tests.hypothesis_strategies.conditions import (
    sub_selection_conditions,
    threshold_conditions,
)

_nonzero_finite_floats = finite_floats().filter(lambda x: x != 0.0)
"""A strategy producing non-zero finite floats."""


@st.composite
def discrete_selection_constraints(
    draw: st.DrawFn, parameters: list[DiscreteParameter] | None = None
):
    """Generate :class:`baybe.constraints.discrete.DiscreteSelectionConstraint`."""
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
    exclude = draw(st.booleans())
    return DiscreteSelectionConstraint(
        parameter_names, conditions, combiner, exclude=exclude
    )


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

    exclude = draw(st.booleans())
    return DiscreteDependenciesConstraint(
        parameter_names, conditions, affected_parameter_names, exclude=exclude
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

    exclude = draw(st.booleans())
    return DiscretePermutationInvarianceConstraint(
        parameter_names, dependencies, exclude=exclude
    )


@st.composite
def discrete_linear_constraints(
    draw: st.DrawFn,
    parameter_names: list[str] | None = None,
):
    """Generate :class:`baybe.constraints.discrete.DiscreteLinearConstraint`."""
    if parameter_names is None:
        params = draw(st.lists(st.text(), unique=True, min_size=1))
    else:
        assert len(parameter_names) > 0
        assert len(parameter_names) == len(set(parameter_names))
        params = parameter_names

    operator = draw(st.sampled_from(list(_threshold_operators.keys())))
    rhs = draw(finite_floats())
    exclude = draw(st.booleans())

    # Optionally add tolerance for tolerance-enabled operators
    tolerance = None
    if operator in _valid_tolerance_operators:
        tolerance = draw(st.one_of(st.none(), finite_floats().filter(lambda x: x > 0)))

    # Optionally add coefficients
    if draw(st.booleans()):
        coefficients = draw(st.tuples(*([_nonzero_finite_floats] * len(params))))
        return DiscreteLinearConstraint(
            params,
            operator,
            coefficients,
            rhs=rhs,
            tolerance=tolerance,
            exclude=exclude,
        )
    return DiscreteLinearConstraint(
        params, operator, rhs=rhs, tolerance=tolerance, exclude=exclude
    )


@st.composite
def discrete_product_constraints(
    draw: st.DrawFn,
    parameter_names: list[str] | None = None,
):
    """Generate :class:`baybe.constraints.discrete.DiscreteProductConstraint`."""
    if parameter_names is None:
        params = draw(st.lists(st.text(), unique=True, min_size=1))
    else:
        assert len(parameter_names) > 0
        assert len(parameter_names) == len(set(parameter_names))
        params = parameter_names

    operator = draw(st.sampled_from(list(_threshold_operators.keys())))
    rhs = draw(finite_floats())
    exclude = draw(st.booleans())

    # Optionally add tolerance for tolerance-enabled operators
    tolerance = None
    if operator in _valid_tolerance_operators:
        tolerance = draw(st.one_of(st.none(), finite_floats().filter(lambda x: x > 0)))

    return DiscreteProductConstraint(
        params, operator=operator, rhs=rhs, tolerance=tolerance, exclude=exclude
    )


@st.composite
def discrete_repetition_constraints(
    draw: st.DrawFn, parameter_names: list[str] | None = None
):
    """Generate :class:`baybe.constraints.discrete.DiscreteRepetitionConstraint`."""
    if parameter_names is None:
        params = draw(st.lists(st.text(), unique=True, min_size=2))
    else:
        assert len(parameter_names) >= 2
        params = parameter_names

    n_max = draw(st.integers(min_value=1, max_value=len(params) - 1))
    exclude = draw(st.booleans())
    return DiscreteRepetitionConstraint(
        params, n_max_repetitions=n_max, exclude=exclude
    )


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

    coefficients = draw(st.tuples(*([_nonzero_finite_floats] * len(parameter_names))))
    rhs = draw(finite_floats())
    is_interpoint = draw(st.booleans())

    # Optionally add the operator
    operators = operators or ["=", ">=", "<="]
    operator = draw(st.sampled_from(operators))

    return ContinuousLinearConstraint(
        parameter_names, operator, coefficients, rhs, is_interpoint
    )


continuous_linear_equality_constraints = partial(
    continuous_linear_constraints, operators=["="]
)
"""Generate linear equality constraints."""

continuous_linear_inequality_constraints = partial(
    continuous_linear_constraints, operators=[">=", "<="]
)
"""Generate linear inequality constraints."""

constraints = st.one_of(
    [
        discrete_selection_constraints(),
        discrete_dependencies_constraints(),
        discrete_permutation_invariance_constraints(),
        discrete_linear_constraints(),
        discrete_product_constraints(),
        discrete_repetition_constraints(),
        continuous_linear_equality_constraints(),
        continuous_linear_inequality_constraints(),
    ]
)
"""A strategy that generates constraints."""
