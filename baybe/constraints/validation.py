"""Validation functionality for constraints."""

from collections.abc import Collection, Sequence
from itertools import combinations

from baybe.constraints.base import Constraint
from baybe.constraints.continuous import ContinuousCardinalityConstraint
from baybe.constraints.discrete import (
    DiscreteDependenciesConstraint,
)
from baybe.parameters import NumericalContinuousParameter
from baybe.parameters.base import Parameter


def validate_constraints(  # noqa: DOC101, DOC103
    constraints: Collection[Constraint], parameters: Collection[Parameter]
) -> None:
    """Assert that a given collection of constraints is valid.

    Raises:
        ValueError: If there is more than one
            :class:`baybe.constraints.discrete.DiscreteDependenciesConstraint` declared.
        ValueError: If any two continuous cardinality constraints have an overlapping
            parameter set.
        ValueError: If any constraint contains an invalid parameter name.
        ValueError: If any continuous constraint includes a discrete parameter.
        ValueError: If any discrete constraint includes a continuous parameter.
        ValueError: If any discrete constraint that is valid only for numerical
            discrete parameters includes non-numerical discrete parameters.
        ValueError: If the bounds of any parameter in a cardinality constraint does
            not cover zero.
    """
    if sum(isinstance(itm, DiscreteDependenciesConstraint) for itm in constraints) > 1:
        raise ValueError(
            f"There is only one {DiscreteDependenciesConstraint.__name__} allowed. "
            f"Please specify all dependencies in one single constraint."
        )

    validate_cardinality_constraints_are_nonoverlapping(
        [con for con in constraints if isinstance(con, ContinuousCardinalityConstraint)]
    )

    param_names_all = [p.name for p in parameters]
    param_names_discrete = [p.name for p in parameters if p.is_discrete]
    param_names_continuous = [p.name for p in parameters if p.is_continuous]
    param_names_non_numerical = [p.name for p in parameters if not p.is_numerical]
    params_continuous: list[NumericalContinuousParameter] = [
        p for p in parameters if isinstance(p, NumericalContinuousParameter)
    ]

    for constraint in constraints:
        if not all(p in param_names_all for p in constraint.parameters):
            raise ValueError(
                f"You are trying to create a constraint with at least one parameter "
                f"name that does not exist in the list of defined parameters. "
                f"Parameter list of the affected constraint: {constraint.parameters}"
            )

        if constraint.is_continuous and any(
            p in param_names_discrete for p in constraint.parameters
        ):
            raise ValueError(
                f"You are trying to initialize a continuous constraint over a "
                f"parameter that is discrete. Parameter list of the affected "
                f"constraint: {constraint.parameters}"
            )

        if constraint.is_discrete and any(
            p in param_names_continuous for p in constraint.parameters
        ):
            raise ValueError(
                f"You are trying to initialize a discrete constraint over a parameter "
                f"that is continuous. Parameter list of the affected constraint: "
                f"{constraint.parameters}"
            )

        if constraint.numerical_only and any(
            p in param_names_non_numerical for p in constraint.parameters
        ):
            raise ValueError(
                f"You are trying to initialize a constraint of type "
                f"'{constraint.__class__.__name__}', which is valid only for numerical "
                f"discrete parameters, over a non-numerical parameter. "
                f"Parameter list of the affected constraint: {constraint.parameters}."
            )

        if isinstance(constraint, ContinuousCardinalityConstraint):
            validate_parameters_bounds_in_cardinality_constraint(
                params_continuous, constraint
            )


def validate_cardinality_constraints_are_nonoverlapping(
    constraints: Collection[ContinuousCardinalityConstraint],
) -> None:
    """Validate that cardinality constraints are non-overlapping.

    Args:
        constraints: A collection of continuous cardinality constraints.

    Raises:
        ValueError: If any two continuous cardinality constraints have an overlapping
            parameter set.
    """
    for c1, c2 in combinations(constraints, 2):
        if (s1 := set(c1.parameters)).intersection(s2 := set(c2.parameters)):
            raise ValueError(
                f"Constraints of type `{ContinuousCardinalityConstraint.__name__}` "
                f"cannot share the same parameters. Found the following overlapping "
                f"parameter sets: {s1}, {s2}."
            )


def validate_parameters_bounds_in_cardinality_constraint(
    parameters: Sequence[NumericalContinuousParameter],
    constraint: ContinuousCardinalityConstraint,
) -> None:
    """Validate that the bounds of all parameters in a cardinality constraint cover
    zero.

    Args:
        parameters: A collection of continuous numerical parameters.
        constraint: A continuous cardinality constraint.

    Raises:
        ValueError: If the bounds of any parameter of a constraint does not cover zero.
    """  # noqa D205
    param_names = [p.name for p in parameters]
    for param_in_constraint in constraint.parameters:
        # Note that this implementation checks implicitly that all constraint
        # parameters must be included in the list of parameters. Otherwise Runtime
        # error occurs.
        if (
            param := parameters[param_names.index(param_in_constraint)]
        ) and not param.is_in_range(0.0):
            raise ValueError(
                f"The bounds of all parameters in a constraint of type "
                f"`{ContinuousCardinalityConstraint.__name__}` must cover "
                f"zero. Either correct the parameter ({param}) bounds:"
                f" {param.bounds=} or remove the parameter {param} from the "
                f"{constraint=} and update the minimum/maximum cardinality "
                f"accordingly."
            )
