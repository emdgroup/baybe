"""Numerical parameters."""

import gc
from functools import cached_property
from typing import Any, ClassVar

import cattrs
import numpy as np
import pandas as pd
from attrs import define, field
from attrs.validators import min_len

from baybe.exceptions import NumericalUnderflowError
from baybe.parameters.base import ContinuousParameter, DiscreteParameter
from baybe.parameters.validation import validate_is_finite, validate_unique_values
from baybe.utils.interval import InfiniteIntervalError, Interval, convert_bounds
from baybe.utils.numerical import DTypeFloatNumpy


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

    @property
    def values(self) -> tuple:  # noqa: D102
        # See base class.
        return tuple(DTypeFloatNumpy(itm) for itm in self._values)

    @cached_property
    def comp_df(self) -> pd.DataFrame:  # noqa: D102
        # See base class.
        comp_df = pd.DataFrame(
            {self.name: self.values}, index=self.values, dtype=DTypeFloatNumpy
        )
        return comp_df

    def is_in_range(self, item: float) -> bool:  # noqa: D102
        # See base class.
        differences_acceptable = [
            np.abs(val - item) <= self.tolerance for val in self.values
        ]
        return any(differences_acceptable)


@define(frozen=True, slots=False)
class NumericalContinuousParameter(ContinuousParameter):
    """Class for continuous numerical parameters."""

    # class variables
    is_numerical: ClassVar[bool] = True
    # See base class.

    # object variables
    bounds: Interval = field(default=None, converter=convert_bounds)
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

    def is_in_range(self, item: float) -> bool:  # noqa: D102
        # See base class.

        return self.bounds.contains(item)

    @property
    def comp_rep_columns(self) -> tuple[str, ...]:  # noqa: D102
        # See base class.
        return (self.name,)

    def summary(self) -> dict:  # noqa: D102
        # See base class.
        param_dict = dict(
            Name=self.name,
            Type=self.__class__.__name__,
            Lower_Bound=self.bounds.lower,
            Upper_Bound=self.bounds.upper,
        )
        return param_dict


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
