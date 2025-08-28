"""Continuous constraints."""

from __future__ import annotations

import gc
import math
from collections.abc import Collection, Iterator, Sequence
from itertools import combinations
from math import comb
from typing import TYPE_CHECKING, Any

import cattrs
import numpy as np
from attrs import define, field
from attrs.validators import deep_iterable, gt, in_, lt

from baybe.constraints.base import (
    CardinalityConstraint,
    ContinuousConstraint,
    ContinuousNonlinearConstraint,
)
from baybe.parameters import NumericalContinuousParameter
from baybe.utils.interval import Interval
from baybe.utils.numerical import DTypeFloatNumpy
from baybe.utils.validation import finite_float

if TYPE_CHECKING:
    from torch import Tensor

_valid_linear_constraint_operators = ["=", ">=", "<="]


@define
class ContinuousLinearConstraint(ContinuousConstraint):
    """Class for continuous linear constraints.

    Continuous linear constraints use parameter lists and coefficients to define
    in-/equality constraints over a continuous parameter space.
    """

    # object variables
    operator: str = field(validator=in_(_valid_linear_constraint_operators))
    """Defines the operator used in the equation. Internally this will negate rhs and
    coefficients for `<=`."""

    coefficients: tuple[float, ...] = field(
        converter=lambda x: cattrs.structure(x, tuple[float, ...]),
        validator=deep_iterable(member_validator=finite_float),
    )
    """In-/equality coefficient for each entry in ``parameters``."""

    rhs: float = field(default=0.0, converter=float, validator=finite_float)
    """Right-hand side value of the in-/equality."""

    @coefficients.validator
    def _validate_coefficients(  # noqa: DOC101, DOC103
        self, _: Any, coefficients: Sequence[float]
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
    def _default_coefficients(self) -> tuple[float, ...]:
        """Return equal weight coefficients as default."""
        return (1.0,) * len(self.parameters)

    @property
    def _multiplier(self) -> float:
        """The internal multiplier for rhs and coefficients."""
        return -1.0 if self.operator == "<=" else 1.0

    @property
    def is_eq(self):
        """Whether this constraint models an equality (assumed inequality otherwise)."""
        return self.operator == "="

    def _drop_parameters(
        self, parameter_names: Collection[str]
    ) -> ContinuousLinearConstraint:
        """Create a copy of the constraint with certain parameters removed.

        Args:
            parameter_names: The names of the parameter to be removed.

        Returns:
            The reduced constraint.
        """
        parameters = [p for p in self.parameters if p not in parameter_names]
        coefficients = tuple(
            c
            for p, c in zip(self.parameters, self.coefficients, strict=True)
            if p not in parameter_names
        )
        return ContinuousLinearConstraint(
            parameters, self.operator, coefficients, self.rhs
        )

    def to_botorch(
        self, parameters: Sequence[NumericalContinuousParameter], idx_offset: int = 0
    ) -> tuple[Tensor, Tensor, float]:
        """Cast the constraint in a format required by botorch.

        Used in calling ``optimize_acqf_*`` functions, for details see
        :func:`botorch.optim.optimize.optimize_acqf`

        Args:
            parameters: The parameter objects of the continuous space.
            idx_offset: Offset to the provided parameter indices.

        Returns:
            The tuple required by botorch.
        """
        import torch

        from baybe.utils.torch import DTypeFloatTorch

        param_names = [p.name for p in parameters]
        param_indices = [
            param_names.index(p) + idx_offset
            for p in self.parameters
            if p in param_names
        ]

        return (
            torch.tensor(param_indices),
            torch.tensor(
                [self._multiplier * c for c in self.coefficients], dtype=DTypeFloatTorch
            ),
            np.asarray(self._multiplier * self.rhs, dtype=DTypeFloatNumpy).item(),
        )


@define
class ContinuousCardinalityConstraint(
    CardinalityConstraint, ContinuousNonlinearConstraint
):
    """Class for continuous cardinality constraints."""

    relative_threshold: float = field(
        default=1e-3, converter=float, validator=[gt(0.0), lt(1.0)]
    )
    """A relative threshold for determining if a value is considered zero.

    The threshold is translated into an asymmetric open interval around zero via
    :meth:`get_absolute_thresholds`.

    **Note:** The interval induced by the threshold is considered **open** because
    numerical routines that optimize parameter values on the complementary set (i.e. the
    value range considered "nonzero") may push the numerical value exactly to the
    interval boundary, which should therefore also be considered "nonzero".
    """

    @property
    def n_inactive_parameter_combinations(self) -> int:
        """The number of possible inactive parameter combinations."""
        return sum(
            comb(len(self.parameters), n_inactive_parameters)
            for n_inactive_parameters in self._inactive_set_sizes()
        )

    def _inactive_set_sizes(self) -> range:
        """Get all possible sizes of inactive parameter sets."""
        return range(
            len(self.parameters) - self.max_cardinality,
            len(self.parameters) - self.min_cardinality + 1,
        )

    def inactive_parameter_combinations(self) -> Iterator[frozenset[str]]:
        """Get an iterator over all possible combinations of inactive parameters."""
        for n_inactive_parameters in self._inactive_set_sizes():
            yield from combinations(self.parameters, n_inactive_parameters)

    def sample_inactive_parameters(self, batch_size: int = 1) -> list[set[str]]:
        """Sample sets of inactive parameters according to the cardinality constraints.

        Args:
            batch_size: The number of parameter sets to be sampled.

        Returns:
            A list of sampled inactive parameter sets, where each set holds the
            corresponding parameter names.
        """
        # The number of possible parameter configuration per set cardinality
        n_configurations_per_cardinality = [
            math.comb(len(self.parameters), n)
            for n in range(self.min_cardinality, self.max_cardinality + 1)
        ]

        # Probability of each set cardinality under the assumption that all possible
        # inactive parameter sets are equally likely
        probabilities = n_configurations_per_cardinality / np.sum(
            n_configurations_per_cardinality
        )

        # Sample the number of active/inactive parameters
        n_active_params = np.random.choice(
            np.arange(self.min_cardinality, self.max_cardinality + 1),
            batch_size,
            p=probabilities,
        )
        n_inactive_params = len(self.parameters) - n_active_params

        # Sample the inactive parameters
        inactive_params = [
            set(np.random.choice(self.parameters, n_inactive, replace=False))
            for n_inactive in n_inactive_params
        ]

        return inactive_params

    def get_absolute_thresholds(self, bounds: Interval, /) -> Interval:
        """Get the absolute thresholds for a given interval.

        Turns the relative threshold of the constraint into absolute thresholds
        for the considered interval. That is, for a given interval ``(a, b)`` with
        ``a <= 0`` and ``b >= 0``, the method returns the interval ``(r*a, r*b)``,
        where ``r`` is the relative threshold defined by the constraint.

        Args:
            bounds: The specified interval.

        Returns:
            The absolute thresholds represented as an interval.

        Raises:
            ValueError: When the specified interval does not contain zero.
        """
        if not bounds.contains(0.0):
            raise ValueError(
                f"The specified interval must contain zero. Given: {bounds.to_tuple()}."
            )

        return Interval(
            lower=self.relative_threshold * bounds.lower,
            upper=self.relative_threshold * bounds.upper,
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
