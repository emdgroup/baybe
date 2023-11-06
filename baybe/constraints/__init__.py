"""BayBE constraints."""

from baybe.constraints.base import Constraint
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
from baybe.constraints.validation import validate_constraints

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
    "validate_constraints",
    "DISCRETE_CONSTRAINTS_FILTERING_ORDER",
]
