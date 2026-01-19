"""Fidelity parameters."""

from functools import cached_property
from typing import Any, ClassVar, cast

import cattrs
import pandas as pd
from attrs import Converter, define, field, fields
from attrs.validators import and_, deep_iterable, ge, instance_of, le, min_len
from typing_extensions import override

from baybe.parameters.base import DiscreteParameter
from baybe.parameters.enum import CategoricalEncoding
from baybe.parameters.validation import (
    validate_contains_one,
    validate_is_finite,
    validate_unique_values,
)
from baybe.utils.conversion import expand_scalar_progression
from baybe.utils.numerical import DTypeFloatNumpy


def _convert_zetas(value, self) -> tuple[float, ...]:
    seq_len = len(self._values)
    if isinstance(value, (int, float)):
        expanded = expand_scalar_progression(value, seq_len)
    else:
        expanded = cattrs.structure(value, tuple[float, ...])
    return expanded


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

    _costs: tuple[float, ...] = field(
        alias="costs",
        converter=lambda x: cattrs.structure(x, tuple[float, ...]),
        validator=[
            validate_is_finite,
            deep_iterable(member_validator=ge(0.0)),
        ],
    )
    """The costs associated with querying the parameter at each value."""

    # TODO: Handle other kinds of assumption about the relationship between fidelities.
    # _zeta currently takes the role of the discrepancy parameter in MF-GP-UCB
    # (Kandasamy et al, 2017) but other parameters may be needed for more general
    # multi-fidelity approaches which use a CategoricalFidelityParameter.
    # TODO: Add an argument for adaptive learning of fidelity discrepancy.
    # To be added in an upcoming pull request.
    _zeta: tuple[float, ...] = field(
        alias="zeta",
        converter=Converter(  # type: ignore
            _convert_zetas,
            takes_self=True,
        ),
        validator=(
            validate_is_finite,
            deep_iterable(member_validator=ge(0.0)),
        ),
    )
    """The maximum discrepancy from target (high) fidelity at any design choice.

    Either a tuple of positive values, , one for each fidelity, equal to 0 at the
    highest fidelity, or a scalar specifying that the first fidelity input into
    'values' has discrepancy 0 (the highest fidelity), the next have discrepancy
    'zeta', 2*'zeta' and so on."""

    _highest_fidelity: str | None = field(
        alias="highest_fidelity",
        validator=instance_of(str),
        default=None,
    )
    """The name of the highest fidelity value. Determined by 'zeta' if not given."""

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
                f"and '{fields(type(self))._values.alias}'"
                f"different in '{self.name}'."
            )

    @_zeta.validator
    def _validate_zeta(  # noqa: DOC101, DOC103
        self, _: Any, value: tuple[float, ...]
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

    @_highest_fidelity.validator
    def _validate_highest_fidelity(  # noqa: DOC101, DOC103
        self, _: Any, target_value: str
    ):
        if target_value not in self._values:
            raise ValueError(
                f"'{fields(type(self))._highest_fidelity.alias}' {target_value} is "
                f"not in '{fields(type(self))._values.alias}' in '{self.name}'."
            )

        target_idx = self._values.index(target_value)

        if isinstance(self._zeta, tuple):
            if self._zeta[target_idx] != 0:
                raise ValueError(
                    f"'{fields(type(self))._highest_fidelity.alias}' must have "
                    f"'{fields(type(self))._zeta.alias}' value of '0' in the "
                    f"fidelity parameter '{self.name}'."
                )

        elif isinstance(self._zeta, float):
            if target_idx != 0:
                raise ValueError(
                    f"When specifying scalar '{fields(type(self))._zeta.alias}', "
                    f"'{fields(type(self))._highest_fidelity.alias}' must be "
                    f"the first name in '{fields(type(self))._values.alias}' so it "
                    f"has '{fields(type(self))._zeta.alias} = 0' in '{self.name}'."
                )

    @override
    @property
    def values(self) -> tuple:
        """The fidelity values of the parameter, sorted lexicographically."""
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
        return tuple(self._costs[f] for f in sorted_fidelities)

    @property
    def zeta(self) -> tuple:
        """The fidelity discrepancies of the parameter, sorted according to values."""
        if isinstance(self._zeta, float):
            fids = range(len(self._values))
            zeta_tup = tuple(f * self._zeta for f in fids)
        else:
            zeta_tup = self._zeta

        sorted_fidelities = sorted(
            range(len(self._values)), key=lambda i: self._values[i]
        )
        return tuple(zeta_tup[f] for f in sorted_fidelities)

    @override
    @cached_property
    def comp_df(self) -> pd.DataFrame:
        comp_df = pd.DataFrame(
            range(len(self.values)), dtype=DTypeFloatNumpy, columns=[self.name]
        )

        return comp_df

    @property
    def highest_fidelity(self) -> str:
        """Highest fidelity value, set manually or otherwise by ``zeta``."""
        if self._highest_fidelity is None:
            return self.values[self.zeta.index(0.0)]
        else:
            return self._highest_fidelity

    @property
    def highest_fidelity_comp(self) -> int:
        """Integer encoding value of the highest fidelity."""
        return cast(int, self.comp_df.loc[self.highest_fidelity, self.name])


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
