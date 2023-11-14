"""Numerical parameters."""

from functools import cached_property
from typing import Any, ClassVar, Tuple

import cattrs
import numpy as np
import pandas as pd
from attr import define, field
from attr.validators import min_len
from scipy.spatial.distance import pdist

from baybe.parameters.base import DiscreteParameter, Parameter
from baybe.parameters.validation import validate_unique_values
from baybe.utils import InfiniteIntervalError, Interval, convert_bounds


@define(frozen=True, slots=False)
class NumericalDiscreteParameter(DiscreteParameter):
    """Parameter class for discrete numerical parameters (a.k.a. setpoints)."""

    # class variables
    is_numeric: ClassVar[bool] = True
    # See base class.

    # object variables
    _values: Tuple[float, ...] = field(
        # FIXME[typing]: https://github.com/python-attrs/cattrs/issues/111
        converter=lambda x: cattrs.structure(x, Tuple[float, ...]),  # type: ignore
        # FIXME[typing]: https://github.com/python-attrs/attrs/issues/1197
        validator=[
            min_len(2),
            validate_unique_values,  # type: ignore
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
        # NOTE: computing all pairwise distances can be avoided if we ensure that the
        #   values are ordered (which is currently not the case)
        dists = pdist(np.asarray(self.values).reshape(-1, 1))
        max_tol = dists.min() / 2.0

        if tolerance >= max_tol:
            raise ValueError(
                f"Parameter {self.name} is initialized with tolerance "
                f"{tolerance} but due to the values {self.values} a "
                f"maximum tolerance of {max_tol} is suggested to avoid ambiguity."
            )

    @property
    def values(self) -> tuple:  # noqa: D102
        # See base class.
        return self._values

    @cached_property
    def comp_df(self) -> pd.DataFrame:  # noqa: D102
        # See base class.
        comp_df = pd.DataFrame({self.name: self.values}, index=self.values)
        return comp_df

    def is_in_range(self, item: float) -> bool:  # noqa: D102
        # See base class.
        differences_acceptable = [
            np.abs(val - item) <= self.tolerance for val in self.values
        ]
        return any(differences_acceptable)


@define(frozen=True, slots=False)
class NumericalContinuousParameter(Parameter):
    """Parameter class for continuous numerical parameters."""

    # class variables
    is_numeric: ClassVar[bool] = True
    # See base class.

    is_discrete: ClassVar[bool] = False
    # See base class.

    # object variables
    bounds: Interval = field(default=None, converter=convert_bounds)
    """The bounds of the parameter."""

    @bounds.validator
    def _validate_bounds(self, _: Any, value: Interval) -> None:  # noqa: DOC101, DOC103
        """Validate bounds.

        Raises:
            InfiniteIntervalError: If the provided interval is infinite.
        """
        if not value.is_finite:
            raise InfiniteIntervalError(
                f"You are trying to initialize a parameter with an infinite interval "
                f"of {value.to_tuple()}. Infinite intervals for parameters are "
                f"currently not supported."
            )

    def is_in_range(self, item: float) -> bool:  # noqa: D102
        # See base class.

        return self.bounds.contains(item)
