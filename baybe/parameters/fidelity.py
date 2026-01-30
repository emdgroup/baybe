"""Fidelity parameters."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from numbers import Real
from typing import Any, ClassVar

import cattrs
import pandas as pd
from attrs import Attribute, Converter, define, field, fields
from attrs.validators import and_, deep_iterable, ge, le, min_len
from typing_extensions import override

from baybe.parameters.base import DiscreteParameter
from baybe.parameters.enum import CategoricalEncoding
from baybe.parameters.validation import (
    validate_contains_exactly_one_zero,
    validate_contains_one,
    validate_is_finite,
    validate_unique_values,
)
from baybe.utils.numerical import DTypeFloatNumpy


def _convert_zeta(
    value: Real | Sequence[Real], self: CategoricalFidelityParameter
) -> tuple[float, ...]:
    """Convert zeta input (sequence or scalar) into a tuple of floats."""
    if isinstance(value, Real):
        seq_len = len(self._values)
        return tuple(i * value for i in range(seq_len))

    return cattrs.structure(value, tuple[float, ...])


@define(frozen=True, slots=False)
class CategoricalFidelityParameter(DiscreteParameter):
    """Parameter class for categorical fidelity parameters."""

    encoding: CategoricalEncoding = field(init=False, default=CategoricalEncoding.INT)
    # See base class.

    _values: tuple[str, ...] = field(
        alias="values",
        converter=lambda x: cattrs.structure(x, tuple[str, ...]),
        validator=[
            min_len(2),
            validate_unique_values,  # type: ignore
        ],
    )
    # See base class.

    costs: tuple[float, ...] = field(
        converter=lambda x: cattrs.structure(x, tuple[float, ...]),
        validator=[validate_is_finite, deep_iterable(member_validator=ge(0.0))],
    )
    """The costs associated with querying the parameter at each value."""

    zeta: tuple[float, ...] = field(
        converter=Converter(_convert_zeta, takes_self=True),
        validator=(
            validate_is_finite,
            deep_iterable(member_validator=ge(0.0)),
            validate_contains_exactly_one_zero,
        ),
    )
    """The maximum discrepancy from target (high) fidelity at any design choice.

    Either a tuple of positive values, , one for each fidelity, equal to 0 at the
    highest fidelity, or a scalar specifying that the first fidelity input into
    'values' has discrepancy 0 (the highest fidelity), the next have discrepancy
    'zeta', 2*'zeta' and so on."""

    @costs.validator
    def _validate_cost_length(  # noqa: DOC101, DOC103
        self, _: Attribute, costs: tuple[float, ...]
    ) -> None:
        """Validate that there is one cost per fidelity parameter.

        Raises:
            ValueError: If 'costs' and 'values' have different lengths.
        """
        if len(costs) != len(self._values):
            raise ValueError(
                f"Length of '{fields(type(self))._costs.alias}'"
                f"and '{fields(type(self))._values.alias}'"
                f"different in '{self.name}'."
            )

    @zeta.validator
    def _validate_zeta(  # noqa: DOC101, DOC103
        self, _: Attribute, value: tuple[float, ...]
    ) -> None:
        """Validate instance attribute ``zeta``.

        Raises:
            ValueError: If ``zeta`` and ``values`` are different lengths.
        """
        if len(value) != len(self._values):
            raise ValueError(
                f"Tuples '{fields(type(self))._zeta.alias}'"
                f"and '{fields(type(self))._values.alias}' are"
                f"different lengths in '{self.name}'."
            )

    def __attrs_post_init__(self) -> None:
        """Sort attribute values according to lexographic fidelity values."""
        idx = sorted(range(len(self._values)), key=lambda i: self._values[i])
        object.__setattr__(self, "_values", tuple(self._values[i] for i in idx))
        object.__setattr__(self, "costs", tuple(self.costs[i] for i in idx))
        object.__setattr__(self, "zeta", tuple(self.zeta[i] for i in idx))

    @override
    @property
    def values(self) -> tuple[float, ...]:
        return self._values

    @override
    @cached_property
    def comp_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            range(len(self.values)), dtype=DTypeFloatNumpy, columns=[self.name]
        )


@define(frozen=True, slots=False)
class NumericalDiscreteFidelityParameter(DiscreteParameter):
    """Parameter class for numerical discrete fidelity parameters.

    Fidelity values are floats in the range [0, 1], including 1 (target fidelity).
    """

    # class variables
    is_numerical: ClassVar[bool] = True
    # See base class.

    _values: tuple[float, ...] = field(
        alias="values",
        converter=lambda x: cattrs.structure(x, tuple[float, ...]),
        validator=[
            min_len(2),
            validate_unique_values,  # type: ignore
            validate_is_finite,
            deep_iterable(member_validator=and_(ge(0.0), le(1.0))),
            validate_contains_one,
        ],
    )
    # See base class.

    _costs: tuple[float, ...] = field(
        alias="costs",
        converter=lambda x: cattrs.structure(x, tuple[float, ...]),
        validator=[
            min_len(2),
            validate_is_finite,
            deep_iterable(member_validator=ge(0.0)),
        ],
    )
    """The costs associated with querying the parameter at each value."""

    @_costs.validator
    def _validate_cost_length(  # noqa: DOC101, DOC103
        self, _: Any, value: tuple[float, ...]
    ) -> None:
        """Validate that there is one cost per fidelity parameter.

        Raises:
            ValueError: If 'costs' and 'values' have different lengths.
        """
        if len(value) != len(self._values):
            raise ValueError(
                f"Length of '{fields(type(self))._costs.alias}'"
                f"and '{fields(type(self))._values.alias}' different in '{self.name}'."
            )

    @override
    @property
    def values(self) -> tuple:
        """The fidelity values of the parameter, sorted in numerical order."""
        sorted_fidelities = sorted(
            range(len(self._values)), key=lambda i: self._values[i]
        )
        return tuple(self._values[f] for f in sorted_fidelities)

    @property
    def costs(self) -> tuple:
        """The fidelity costs of the parameter, sorted according to values."""
        sorted_fidelities = sorted(
            range(len(self._values)), key=lambda i: self._values[i]
        )
        return tuple(DTypeFloatNumpy(self._costs[f]) for f in sorted_fidelities)

    @override
    @cached_property
    def comp_df(self) -> pd.DataFrame:
        comp_df = pd.DataFrame(
            {self.name: self.values}, index=self.values, dtype=DTypeFloatNumpy
        )
        return comp_df
