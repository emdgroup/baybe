"""Fidelity parameters."""

from functools import cached_property
from typing import Any, ClassVar, cast

import cattrs
import numpy as np
import pandas as pd
from attrs import define, field
from attrs.validators import and_, deep_iterable, ge, le, min_len
from typing_extensions import override

from baybe.parameters.base import DiscreteParameter
from baybe.parameters.enum import CategoricalEncoding
from baybe.parameters.validation import (
    validate_contains_one,
    validate_is_finite,
    validate_unique_values,
)
from baybe.utils.numerical import DTypeFloatNumpy


@define(frozen=True, slots=False)
class CategoricalFidelityParameter(DiscreteParameter):
    """Parameter class for categorical fidelity parameters."""

    # class variables
    is_numerical: ClassVar[bool] = True
    # See base class.

    # object variables
    encoding: CategoricalEncoding | None = field(
        init=False,
        default=None,
    )
    # See base class.

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
        # FIXME[typing]: https://github.com/python-attrs/cattrs/issues/111
        converter=lambda x: cattrs.structure(x, tuple[float, ...]),
        # FIXME[typing]: https://github.com/python-attrs/attrs/issues/1197
        validator=[
            min_len(2),
            validate_is_finite,
            deep_iterable(ge(0.0)),
        ],
    )
    """The costs associated with querying the parameter at each value."""

    _target_fidelity: str = field(
        alias="target_fidelity",
        converter=str,
    )
    """The column index of the target fidelity value."""

    # TODO Jordan MHS: handle hyperparameters from different acqfs.
    _zeta: tuple[float, ...] | float = field(
        alias="zeta",
        # FIXME[typing]: https://github.com/python-attrs/cattrs/issues/111
        converter=lambda x: (
            None
            if x is None
            else cattrs.structure(x, float)
            if isinstance(x, (float, int))
            else cattrs.structure(x, tuple[float, ...])
        ),
        default=None,
    )
    """The maximum discrepancy from target fidelity at any design choice."""

    @_costs.validator
    def _validate_cost_length(  # noqa: DOC101, DOC103
        self, _: Any, value: tuple[float, ...]
    ) -> None:
        """Validate that ``len(_costs)`` equals ``len(_values)``.

        Raises:
            ValueError: If ``len(_costs) != len(_values)``.
        """
        if len(value) != len(self._values):
            raise ValueError(f"Length of 'costs' and 'values' different in {self.name}")

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
            raise ValueError(f"Value 'zeta' undefined in {self.name}")

        elif isinstance(value, float):
            if not np.isfinite(value):
                raise ValueError(f"Parameter 'zeta' is infinite in {self.name}")
            elif value < 0:
                raise ValueError(f"Parameter 'zeta' is negative in {self.name}")
            else:
                return

        else:
            assert isinstance(self._zeta, tuple)

            if len(self._zeta) != len(self._values):
                raise ValueError(
                    f"Tuples 'zeta' and 'values' are different lengths in {self.name}"
                )

            if any(np.isinf(value)):
                raise ValueError(
                    f"Tuple 'zeta' contains infinite values in {self.name}"
                )

    @_target_fidelity.validator
    def _validate_target_fidelity(  # noqa: DOC101, DOC103
        self, _: Any, target_value: str
    ):
        if target_value not in self._values:
            raise ValueError(
                f"'target_fidelity' {target_value} is not in 'values' in {self.name}"
            )

    @override
    @property
    def values(self) -> tuple:
        """The fidelity values of the parameter."""
        sorted_fidelities = sorted(
            range(len(self._values)), key=lambda i: self._values[i]
        )
        return tuple(self._values[f] for f in sorted_fidelities)

    @property
    def costs(self) -> tuple:
        """The fidelity costs of the parameter."""
        sorted_fidelities = sorted(
            range(len(self._values)), key=lambda i: self._values[i]
        )
        return tuple(self._costs[f] for f in sorted_fidelities)

    @property
    def zeta(self) -> tuple:
        """The fidelity discrepancies of the parameter."""
        if isinstance(self._zeta, float):
            fids = range(len(self._values))
            zeta_tup = tuple((f + 1) * self._zeta for f in fids)

        else:
            zeta_tup = self._zeta

        sorted_fidelities = sorted(
            range(len(self._values)), key=lambda i: self._values[i]
        )
        return tuple(zeta_tup[f] for f in sorted_fidelities)

    @property
    def target_fidelity(self) -> str:
        """Categorical value of the target fidelity."""
        return self._target_fidelity

    @override
    @cached_property
    def comp_df(self) -> pd.DataFrame:
        comp_df = pd.DataFrame(
            range(len(self.values)), dtype=DTypeFloatNumpy, columns=[self.name]
        )

        return comp_df

    @property
    def target_fidelity_comp(self) -> int:
        """Integer encoding value of the target fidelity."""
        return cast(int, self.comp_df.loc[self.target_fidelity, self.name])


@define(frozen=True, slots=False)
class NumericalDiscreteFidelityParameter(DiscreteParameter):
    """Parameter class for numerical discrete fidelity parameters.

    Fidelity values are floats in the range [0, 1], including 1 (target fidelity).
    """

    # class variables
    is_numerical: ClassVar[bool] = True
    # See base class.

    # object variables
    encoding: None = field(init=False, default=None)
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
        # FIXME[typing]: https://github.com/python-attrs/cattrs/issues/111
        converter=lambda x: cattrs.structure(x, tuple[float, ...]),
        # FIXME[typing]: https://github.com/python-attrs/attrs/issues/1197
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
        """Validate that ``len(_costs)`` equals ``len(_values)``.

        Raises:
            ValueError: If ``len(_costs) != len(_values)``.
        """
        if len(value) != len(self._values):
            raise ValueError(f"Length of 'costs' and 'values' different in {self.name}")

    @override
    @property
    def values(self) -> tuple:
        """The fidelity values of the parameter."""
        sorted_fidelities = sorted(
            range(len(self._values)), key=lambda i: self._values[i]
        )
        return tuple(self._values[f] for f in sorted_fidelities)

    @property
    def costs(self) -> tuple:
        """The fidelity costs of the parameter."""
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
