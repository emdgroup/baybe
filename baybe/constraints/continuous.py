"""Continuous constraints."""


from attrs import define, field, validators
from collections.abc import Sequence

from baybe.constraints.base import ContinuousConstraint, ContinuousLinearConstraint
from baybe.parameters import NumericalContinuousParameter


@define
class ContinuousLinearEqualityConstraint(ContinuousLinearConstraint):
    """Class for continuous equality constraints.

    The constraint is defined as ``sum_i[ x_i * c_i ] == rhs``, where x_i are the
    parameter names from ``parameters`` and c_i are the entries of ``coefficients``.
    The constraint is typically fulfilled up to a small numerical tolerance.

    The class has no real content as it only serves the purpose of distinguishing the
    constraints.
    """


@define
class ContinuousLinearInequalityConstraint(ContinuousLinearConstraint):
    """Class for continuous inequality constraints.

    The constraint is defined as ``sum_i[ x_i * c_i ] >= rhs``, where x_i are the
    parameter names from ``parameters`` and c_i are the entries of ``coefficients``.
    If you want to implement a constraint of the form `<=`, multiply ``rhs`` and
    ``coefficients`` by -1. The constraint is typically fulfilled up to a small
    numerical tolerance.

    The class has no real content as it only serves the purpose of
    distinguishing the constraints.
    """


@define
class ContinuousCardinalityConstraint(ContinuousConstraint):
    """Class for continuous cardinality constraints.

    The constraint is defined as ``cardinality[..., x_i, ...] >= cardinality_low´´
    and ``cardinality[..., x_i, ...] <= cardinality_up´´.
    """

    cardinality_low: int = field(default=0, validator=validators.ge(0))
    "The lower limit of cardinality"

    cardinality_up: int = field()
    "The upper limit of cardinality"

    @cardinality_up.default
    def _set_cardinality_up_default(self):
        """Set cardinality_up = len(parameter) by default."""
        return len(self.parameters)

    def __attrs_post_init__(self):
        """Validate."""
        # cardinality_low=cardinality_up implies a fixed cardinality.
        if self.cardinality_low > self.cardinality_up:
            raise ValueError(
                f"The lower bound of cardinality should not be larger "
                f"than the upper bound, but was upper bound = "
                f"{self.cardinality_up}, lower bound ="
                f" {self.cardinality_low}."
            )

        # cardinality_up should be <= len(parameters)
        if self.cardinality_up > len(self.parameters):
            raise ValueError(
                f"The upper bound of cardinality should not exceed the "
                f"number of parameters, but was upper bound ="
                f" {self.cardinality_up}, and len(parameters) ="
                f" {len(self.parameters)}."
            )

        # No cardinality constraints are required in this case
        if self.cardinality_low == 0 and self.cardinality_up == len(self.parameters):
            raise ValueError(
                "No cardinality constraint is required when "
                "0<= cardinality <= len(parameters)."
            )

    def to_botorch(  # noqa: D102
        self, parameters: Sequence[NumericalContinuousParameter], idx_offset: int = 0
    ) -> list[tuple[callable, bool]]:
        # See base class.
        pass
