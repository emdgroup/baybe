"""BayBE constraints."""

from baybe.constraints.base import Constraint, validate_constraints
from baybe.constraints.conditions import (
    Condition,
    SubSelectionCondition,
    ThresholdCondition,
)
from baybe.constraints.continuous import (
    ContinuousConstraint,
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
)
from baybe.constraints.discrete import (
    DISCRETE_CONSTRAINTS_FILTERING_ORDER,
    DiscreteConstraint,
    DiscreteCustomConstraint,
    DiscreteDependenciesConstraint,
    DiscreteExcludeConstraint,
    DiscreteLinkedParametersConstraint,
    DiscreteNoLabelDuplicatesConstraint,
    DiscretePermutationInvarianceConstraint,
    DiscreteProductConstraint,
    DiscreteSumConstraint,
)

__all__ = [
    # --- Base classes --- #
    "Condition",
    "Constraint",
    "ContinuousConstraint",
    "DiscreteConstraint",
    # --- Conditions --- #
    "SubSelectionCondition",
    "ThresholdCondition",
    # --- Continuous constraints ---#
    "ContinuousLinearEqualityConstraint",
    "ContinuousLinearInequalityConstraint",
    # --- Discrete constraints ---#
    "DiscreteCustomConstraint",
    "DiscreteDependenciesConstraint",
    "DiscreteExcludeConstraint",
    "DiscreteLinkedParametersConstraint",
    "DiscreteNoLabelDuplicatesConstraint",
    "DiscretePermutationInvarianceConstraint",
    "DiscreteProductConstraint",
    "DiscreteSumConstraint",
    # --- Other --- #
    "DISCRETE_CONSTRAINTS_FILTERING_ORDER",
    "validate_constraints",
]
