"""BayBE constraints."""

from baybe.constraints.conditions import SubSelectionCondition, ThresholdCondition
from baybe.constraints.continuous import (
    ContinuousCardinalityConstraint,
    ContinuousLinearConstraint,
)
from baybe.constraints.deprecation import (
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
)
from baybe.constraints.discrete import (
    DISCRETE_CONSTRAINTS_FILTERING_ORDER,
    DiscreteCardinalityConstraint,
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
    # --- Conditions --- #
    "SubSelectionCondition",
    "ThresholdCondition",
    # --- Continuous constraints ---#
    "ContinuousLinearConstraint",
    "ContinuousCardinalityConstraint",
    "ContinuousLinearEqualityConstraint",
    "ContinuousLinearInequalityConstraint",
    # --- Discrete constraints ---#
    "DiscreteCardinalityConstraint",
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
