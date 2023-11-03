"""Base classes for all constraints."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, List, Tuple

import pandas as pd
import torch
from attr import define, field
from attr.validators import min_len
from torch import Tensor

from baybe.constraints.conditions import Condition
from baybe.parameters import NumericalContinuousParameter, Parameter
from baybe.utils import (
    DTypeFloatTorch,
    get_base_structure_hook,
    SerialMixin,
    unstructure_base,
)
from baybe.utils.serialization import converter


@define
class Constraint(ABC, SerialMixin):
    """Abstract base class for all constraints.

    Constraints use conditions and chain them together to filter unwanted entries from
    the search space.

    Args:
        parameters: The list of parameters used for the constraint.
    """

    # class variables
    # TODO: it might turn out these are not needed at a later development stage
    eval_during_creation: ClassVar[bool]
    """Class variable encoding whether the condition is evaluated during creation."""
    eval_during_modeling: ClassVar[bool]
    """Class variable encoding whether the condition is evaluated during modeling."""

    # Object variables
    parameters: List[str] = field(validator=min_len(1))

    @parameters.validator
    def _validate_params(  # noqa: DOC101, DOC103
        self, _: Any, params: List[str]
    ) -> None:
        """Validate the parameter list.

        Raises:
            ValueError: If ```params``` contains duplicate values.
        """
        if len(params) != len(set(params)):
            raise ValueError(
                f"The given 'parameters' list must have unique values "
                f"but was: {params}."
            )

    @property
    def is_continuous(self) -> bool:
        """Boolean indicating if this is a constraint over continuous parameters."""
        return isinstance(self, ContinuousConstraint)

    @property
    def is_discrete(self) -> bool:
        """Boolean indicating if this is a constraint over discrete parameters."""
        return isinstance(self, DiscreteConstraint)


@define
class DiscreteConstraint(Constraint, ABC):
    """Abstract base class for discrete constraints.

    Discrete constraints use conditions and chain them together to filter unwanted
    entries from the search space.
    """

    # class variables
    eval_during_creation: ClassVar[bool] = True
    eval_during_modeling: ClassVar[bool] = False

    @abstractmethod
    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """Get the indices of dataframe entries that are invalid under the constraint.

        Args:
            data: A dataframe where each row represents a particular parameter
                combination.

        Returns:
            The dataframe indices of rows where the constraint is violated.
        """


@define
class ContinuousConstraint(Constraint, ABC):
    """Abstract base class for continuous constraints.

    Continuous constraints use parameter lists and coefficients to define in-/equality
    constraints over a continuous parameter space.

    Args:
        parameters: See base class.
        coefficients: In-/equality coefficient for each entry in ```parameters```.
        rhs: Right-hand side value of the in-/equality.
    """

    # class variables
    eval_during_creation: ClassVar[bool] = False
    eval_during_modeling: ClassVar[bool] = True

    # object variables
    coefficients: List[float] = field()
    rhs: float = field(default=0.0)

    @coefficients.validator
    def _validate_coefficients(  # noqa: DOC101, DOC103
        self, _: Any, coefficients: List[float]
    ) -> None:
        """Validate the coefficients.

        Raises:
            ValueError: If the number of coefficients does not match the number of
                parameters.
        """
        if len(self.parameters) != len(coefficients):
            raise ValueError(
                "The given 'coefficients' list must have one floating point entry for "
                "each entry in 'parameters'."
            )

    @coefficients.default
    def _default_coefficients(self):
        """Return equal weight coefficients as default."""
        return [1.0] * len(self.parameters)

    def to_botorch(
        self, parameters: List[NumericalContinuousParameter], idx_offset: int = 0
    ) -> Tuple[Tensor, Tensor, float]:
        """Cast the constraint in a format required by botorch.

        Used in calling ```optimize_acqf_*``` functions, for details see
        https://botorch.org/api/optim.html#botorch.optim.optimize.optimize_acqf

        Args:
            parameters: The parameter objects of the continuous space.
            idx_offset: Offset to the provided parameter indices.

        Returns:
            The tuple required by botorch.
        """
        param_names = [p.name for p in parameters]
        param_indices = [
            param_names.index(p) + idx_offset
            for p in self.parameters
            if p in param_names
        ]

        return (
            torch.tensor(param_indices),
            torch.tensor(self.coefficients, dtype=DTypeFloatTorch),
            self.rhs,
        )


# Register (un-)structure hooks
converter.register_unstructure_hook(Condition, unstructure_base)
converter.register_structure_hook(Condition, get_base_structure_hook(Condition))
converter.register_unstructure_hook(Constraint, unstructure_base)
converter.register_structure_hook(Constraint, get_base_structure_hook(Constraint))


def validate_constraints(  # noqa: DOC101, DOC103
    constraints: List[Constraint], parameters: List[Parameter]
) -> None:
    """Assert that a given collection of constraints is valid.

    Raises:
        ValueError: If there is more than one
            :class:`baybe.constraints.DiscreteDependenciesConstraint` declared.
        ValueError: If any constraint contains an invalid parameter name.
        ValueError: If any continuous constraint includes a discrete parameter.
        ValueError: If any discrete constraint includes a continuous parameter.
    """
    from baybe.constraints.discrete import (  # pylint: disable=import-outside-toplevel
        DiscreteDependenciesConstraint,
    )

    if sum(isinstance(itm, DiscreteDependenciesConstraint) for itm in constraints) > 1:
        raise ValueError(
            f"There is only one {DiscreteDependenciesConstraint.__name__} allowed. "
            f"Please specify all dependencies in one single constraint."
        )

    param_names_all = [p.name for p in parameters]
    param_names_discrete = [p.name for p in parameters if p.is_discrete]
    param_names_continuous = [p.name for p in parameters if not p.is_discrete]
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
