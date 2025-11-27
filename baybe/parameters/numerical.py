"""Numerical parameters."""

import gc
from functools import cached_property
from typing import Any, ClassVar

import cattrs
import numpy as np
import pandas as pd
from attrs import define, field
from attrs.validators import min_len
from attrs.validators import optional as optional_v
from typing_extensions import override

from baybe.exceptions import NumericalUnderflowError
from baybe.parameters.base import ContinuousParameter, DiscreteParameter
from baybe.parameters.validation import validate_is_finite, validate_unique_values
from baybe.utils.interval import InfiniteIntervalError, Interval
from baybe.utils.numerical import DTypeFloatNumpy

import torch

@define(frozen=True, slots=False)
class NumericalDiscreteParameter(DiscreteParameter):
    """Class for discrete numerical parameters (a.k.a. setpoints)."""

    # class variables
    is_numerical: ClassVar[bool] = True
    # See base class.

    # object variables
    # NOTE: The parameter values are assumed to be sorted by the tolerance validator.
    _values: tuple[float, ...] = field(
        alias="values",
        # FIXME[typing]: https://github.com/python-attrs/cattrs/issues/111
        converter=lambda x: sorted(cattrs.structure(x, tuple[float, ...])),  # type: ignore
        # FIXME[typing]: https://github.com/python-attrs/attrs/issues/1197
        validator=[
            min_len(2),
            validate_unique_values,  # type: ignore
            validate_is_finite,
        ],
    )
    """The values the parameter can take."""

    tolerance: float = field(default=0.0)
    """The absolute tolerance used for deciding whether a value is in range. A tolerance
        larger than half the minimum distance between parameter values is not allowed
        because that could cause ambiguity when inputting data points later."""

    @tolerance.validator
    def _validate_tolerance(  # noqa: DOC101, DOC103
        self, _: Any, tolerance: float
    ) -> None:
        """Validate that the given tolerance is safe.

        The tolerance is the allowed experimental uncertainty when
        reading in measured values. A tolerance larger than half the minimum
        distance between parameter values is not allowed because that could cause
        ambiguity when inputting data points later.

        Raises:
            ValueError: If the tolerance is not safe.
        """
        # For zero tolerance, the only left requirement is that all parameter values
        # are distinct, which is already ensured by the corresponding validator.
        if tolerance == 0.0:
            return

        min_dist = np.diff(self._values).min()
        if min_dist == (eps := np.nextafter(0, 1)):
            raise NumericalUnderflowError(
                f"The distance between any two parameter values must be at least "
                f"twice the size of the used floating point resolution of {eps}."
            )

        if tolerance >= (max_tol := min_dist / 2.0):
            raise ValueError(
                f"Parameter '{self.name}' is initialized with tolerance {tolerance} "
                f"but due to the given parameter values {self.values}, the specified "
                f"tolerance must be smaller than {max_tol} to avoid ambiguity."
            )

    @override
    @property
    def values(self) -> tuple:
        return tuple(DTypeFloatNumpy(itm) for itm in self._values)

    @override
    @cached_property
    def comp_df(self) -> pd.DataFrame:
        comp_df = pd.DataFrame(
            {self.name: self.values}, index=self.values, dtype=DTypeFloatNumpy
        )
        return comp_df

    @override
    def is_in_range(self, item: float) -> bool:
        return any(
            Interval(val - self.tolerance, val + self.tolerance).contains(item)
            for val in self.values
        )


@define(frozen=True, slots=False)
class NumericalContinuousParameter(ContinuousParameter):
    """Class for continuous numerical parameters."""

    is_numerical: ClassVar[bool] = True
    # See base class.

    # TODO[typing]: https://github.com/python-attrs/attrs/issues/1435
    bounds: Interval = field(default=None, converter=Interval.create)  # type: ignore[misc]
    """The bounds of the parameter."""

    @bounds.validator
    def _validate_bounds(self, _: Any, value: Interval) -> None:  # noqa: DOC101, DOC103
        """Validate bounds.

        Raises:
            InfiniteIntervalError: If the provided interval is not finite.
        """
        if not value.is_bounded:
            raise InfiniteIntervalError(
                f"You are trying to initialize a parameter with an infinite range "
                f"of {value.to_tuple()}. Infinite intervals for parameters are "
                f"currently not supported."
            )
        if value.is_degenerate:
            raise ValueError(
                "The interval specified by the parameter bounds cannot be degenerate."
            )

    @override
    def is_in_range(self, item: float) -> bool:
        return self.bounds.contains(item)

    @override
    @property
    def comp_rep_columns(self) -> tuple[str]:
        return (self.name,)

    @override
    def summary(self) -> dict:
        param_dict = dict(
            Name=self.name,
            Type=self.__class__.__name__,
            Lower_Bound=self.bounds.lower,
            Upper_Bound=self.bounds.upper,
        )
        return param_dict


@define(frozen=True, slots=False)
class _FixedNumericalContinuousParameter(ContinuousParameter):
    """Parameter class for fixed numerical parameters."""

    is_numeric: ClassVar[bool] = True
    # See base class.

    value: float = field(converter=float)
    """The fixed value of the parameter."""

    @property
    def bounds(self) -> Interval:
        """The value of the parameter as a degenerate interval."""
        return Interval(self.value, self.value)

    @override
    def is_in_range(self, item: float) -> bool:
        return Interval(self.value, self.value).contains(item)

    @override
    @property
    def comp_rep_columns(self) -> tuple[str]:
        return (self.name,)

    @override
    def summary(self) -> dict:
        return dict(
            Name=self.name,
            Type=self.__class__.__name__,
            Value=self.value,
        )

@define(frozen=True, slots=False)
class DiscreteFidelityParameter(NumericalDiscreteParameter):
    """Parameter class for fidelity parameters."""


    # Overriding property to be sorted alongside _costs and _zetas in __attrs_post_init__
    _values: tuple[float, ...] = field(
        alias="values",
        converter=lambda x: cattrs.structure(x, tuple[float, ...]),  # type: ignore
        validator=[
            min_len(2),
            validate_unique_values,  # type: ignore
            validate_is_finite,
        ],
        default=None,
    )

    _costs: tuple[float, ...] = field(
        alias="costs",
        # FIXME[typing]: https://github.com/python-attrs/cattrs/issues/111
        converter=lambda x: cattrs.structure(x, tuple[float, ...]),
        # FIXME[typing]: https://github.com/python-attrs/attrs/issues/1197
        validator=[
            min_len(2),
            validate_unique_values,  # type: ignore
            validate_is_finite,
        ],
        default=None,
    )

    _zetas: tuple[float, ...] | None = field(
        alias="zetas",
        # FIXME[typing]: https://github.com/python-attrs/cattrs/issues/111
        converter=lambda x: None if x is None else cattrs.structure(x, tuple[float, ...]),
        # FIXME[typing]: https://github.com/python-attrs/attrs/issues/1197
        validator=optional_v([
            min_len(2),
            validate_unique_values,  # type: ignore
            validate_is_finite,
        ]),
        default=None,
    )

    def __attrs_post_init__(self):

        super_post = getattr(super(), "__attrs_post_init__", None)
        if callable(super_post):
            super_post()

        object.__setattr__(self, "fid_to_idx", {float(f): int(i) for i, f in enumerate(self._values)})
        object.__setattr__(self, "idx_to_fid", {int(i): float(f) for i, f in enumerate(self._values)})
        
        if self._zetas is None: 
            # TODO Jordan MHS: decide on default zeta parameters in future commit - or a default zetas-setting callable.
            object.__setattr__(self, "_zetas", (0.0,) * len(self._values))

        validate_same_shape(self, "values", self._values, "costs", self._costs)
        validate_same_shape(self, "values", self._values, "zetas", self._costs)
        
        paired_values = sorted(zip(self._values, self._zetas, self._costs), key=lambda t: t[0])
        sorted_values, sorted_zetas, sorted_costs = map(tuple, zip(*paired_values))

        object.__setattr__(self, "_values", sorted_values)
        object.__setattr__(self, "_zetas", sorted_zetas)
        object.__setattr__(self, "_costs", sorted_costs)

    def to_index(self, fidelities: torch.Tensor | tuple) -> torch.Tensor:
        return torch.tensor(
            [self.fid_to_idx[float(f)] for f in fidelities],
            dtype=torch.long
        )

    def to_value(self, indices: torch.Tensor | tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(
            [self.idx_to_fid[int(i)] for i in indices],
            dtype=torch.float64
        )

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
