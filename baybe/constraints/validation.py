"""Validation functionality for constraints."""

from collections.abc import Collection

from baybe.constraints.base import Constraint
from baybe.constraints.discrete import DiscreteDependenciesConstraint
from baybe.constraints.continuous import ContinuousCardinalityConstraint
from baybe.parameters.base import ContinuousParameter, DiscreteParameter, Parameter
from baybe.parameters.base import Parameter


def validate_constraints(  # noqa: DOC101, DOC103
    constraints: Collection[Constraint], parameters: Collection[Parameter]
) -> None:
    """Assert that a given collection of constraints is valid.

    Raises:
        ValueError: If there is more than one
            :class:`baybe.constraints.discrete.DiscreteDependenciesConstraint` declared.
        ValueError: If two continuous cardinality constraints share the continuous
        parameter.
        ValueError: If any constraint contains an invalid parameter name.
        ValueError: If any continuous constraint includes a discrete parameter.
        ValueError: If any discrete constraint includes a continuous parameter.
    """
    if sum(isinstance(itm, DiscreteDependenciesConstraint) for itm in constraints) > 1:
        raise ValueError(
            f"There is only one {DiscreteDependenciesConstraint.__name__} allowed. "
            f"Please specify all dependencies in one single constraint."
        )

    # Any cardinality constraints share the same parameter.
    validate_continuous_cardinality_constraints(constraints)

    # Validate parameter/constraint combination.
    param_names_all = [p.name for p in parameters]
    param_names_discrete = [p.name for p in parameters if p.is_discrete]
    param_names_continuous = [p.name for p in parameters if p.is_continuous]
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


def validate_continuous_cardinality_constraints(constraints: Collection[Constraint]):
    """Validate continuous cardinality constraints.

    Raises:
        ValueError: If two cardinality constraints share the same parameter.
    """
    cardinality_constraints = [con for con in constraints
                               if isinstance(con, ContinuousCardinalityConstraint)]
    param_all = []
    for con in cardinality_constraints:
        if len(set(param_all).intersection(set(con.parameters))) != 0:
            raise ValueError("Cardinality constraints cannot share the same "
                             "parameter.")
        param_all.extend(con.parameters)
