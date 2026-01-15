"""Fidelity parameters."""

from functools import cached_property
from typing import Any, ClassVar, cast

import cattrs
import numpy as np
import pandas as pd
from attrs import define, field, fields
from attrs.validators import and_, deep_iterable, ge, le, min_len
from typing_extensions import override

from baybe.parameters.base import DiscreteParameter
from baybe.parameters.validation import (
    validate_contains_one,
    validate_is_finite,
    validate_unique_values,
)
from baybe.utils.numerical import DTypeFloatNumpy


@define(frozen=True, slots=False)
class CategoricalFidelityParameter(DiscreteParameter):
    """Parameter class for categorical fidelity parameters."""

    _values: tuple[str, ...] = field(
        alias="values",
        converter=lambda x: cattrs.structure(x, tuple[str, ...]),  # type: ignore
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
            min_len(2),
            validate_is_finite,
            deep_iterable(member_validator=ge(0.0)),
        ],
    )
    """The costs associated with querying the parameter at each value."""

    high_fidelity: str = field(
        converter=str,
        default=None,
    )
    """The name of the highest fidelity value."""

    # TODO: Handle other kinds of assumption about the relationship between fidelities.
    # _zeta currently takes the role of the discrepancy parameter in MF-GP-UCB
    # (Kandasamy et al, 2017) but other parameters may be needed for more general
    # multi-fidelity approaches which use a CategoricalFidelityParameter.
    _zeta: tuple[float, ...] | float = field(
        alias="zeta",
        converter=lambda x: (
            cattrs.structure(x, float)
            if isinstance(x, (float, int))
            else cattrs.structure(x, tuple[float, ...])
        ),
        default=None,
    )
    """The maximum discrepancy from target (high) fidelity at any design choice.

    Either a tuple of positive values, one for each fidelity, or a scalar specifying
    that the first fidelity input into 'values' has discrepancy 0 (it is the high
    fidelity), the next have discrepancy 'zeta', 2*'zeta' and so on."""

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
                f"Length of '{fields(type(self))._costs.name}'"
                f"and '{fields(type(self))._values.name}'"
                f"different in '{self.name}'."
            )

    @_zeta.validator
    def _validate_zeta(  # noqa: DOC101, DOC103
        self, _: Any, value: tuple[float, ...] | float
    ) -> None:
        """Validate instance attribute ``zeta``.

        Raises:
            ValueError: If ``zeta`` is ``tuple[float, ...]`` and
                ``len(zeta) != len(self._values)``.
            ValueError: If ``zeta`` contains any negative or infinite values.
        """
        if value is None:
            # Jordan MHS TODO: add adaptive_zeta argument
            # Jordan MHS TODO: handle other kinds of discrepancy param
            # Jordan MHS TODO: make zeta optional if above params defined
            raise ValueError(
                f"Argument '{fields(type(self))._zeta.name}'undefined in '{self.name}'."
            )

        elif isinstance(value, float):
            if not np.isfinite(value):
                raise ValueError(
                    f"Arguemnt '{fields(type(self))._zeta.name}'"
                    f"is infinite in '{self.name}'."
                )
            elif value < 0:
                raise ValueError(
                    f"Parameter '{fields(type(self))._zeta.name}'"
                    f"is negative in '{self.name}'."
                )
            else:
                return

        else:
            assert isinstance(value, tuple)

            if len(value) != len(self._values):
                raise ValueError(
                    f"Tuples '{fields(type(self))._zeta.name}'"
                    f"and '{fields(type(self))._values.name}' are"
                    f"different lengths in '{self.name}'."
                )

            if any(np.isinf(value)):
                raise ValueError(
                    f"Tuple '{fields(type(self))._zeta.name}' contains"
                    f"infinite values in '{self.name}'."
                )

    @high_fidelity.validator
    def _validate_high_fidelity(  # noqa: DOC101, DOC103
        self, _: Any, target_value: str
    ):
        if target_value not in self._values:
            raise ValueError(
                f"'{fields(type(self)).high_fidelity.name}' {target_value}"
                f"is not in '{fields(type(self))._values.name}' in '{self.name}'."
            )

        target_idx = self._values.index(target_value)

        if isinstance(self._zeta, tuple):
            if self._zeta[target_idx] != 0:
                raise ValueError(
                    f"'{fields(type(self)).high_fidelity.name}' cannot have a"
                    f"'{fields(type(self))._zeta.name}' value of '0' in the"
                    f"fidelity parameter '{self.name}'."
                )

        elif isinstance(self._zeta, float):
            if target_idx != 0:
                raise ValueError(
                    f"When specifying scalar '{fields(type(self))._zeta.name}',"
                    f"'{fields(type(self)).high_fidelity.name}' must be the first"
                    f"name in '{fields(type(self))._values.name}' so it has"
                    f"'{fields(type(self))._zeta.name} = 0' in '{self.name}'."
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
    def high_fidelity_comp(self) -> int:
        """Integer encoding value of the target fidelity."""
        return cast(int, self.comp_df.loc[self.high_fidelity, self.name])


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
        converter=lambda x: cattrs.structure(x, tuple[float, ...]),  # type: ignore
        validator=[
            min_len(2),
            validate_unique_values,  # type: ignore
            validate_is_finite,
            deep_iterable(and_(ge(0.0), le(1.0))),
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
            deep_iterable(ge(0.0)),
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
                f"Length of '{fields(type(self))._costs.name}'"
                f"and '{fields(type(self))._values.name}' different in '{self.name}'."
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
