"""Binary targets."""

import gc
import warnings
from typing import TypeAlias

import numpy as np
import pandas as pd
from attrs import define, field
from attrs.validators import instance_of
from typing_extensions import override

from baybe.exceptions import InvalidTargetValueError
from baybe.serialization import SerialMixin
from baybe.targets.base import Target
from baybe.utils.validation import validate_not_nan

ChoiceValue: TypeAlias = bool | int | float | str
"""Types of values that a :class:`BinaryTarget` can take."""

_SUCCESS_VALUE_COMP = 1.0
"""Computational representation of the success value."""

_FAILURE_VALUE_COMP = 0.0
"""Computational representation of the failure value."""


@define(frozen=True)
class BinaryTarget(Target, SerialMixin):
    """Class for binary targets."""

    # FIXME[typing]: https://github.com/python-attrs/attrs/issues/1336

    success_value: ChoiceValue = field(
        default=True,
        validator=[instance_of(ChoiceValue), validate_not_nan],  # type: ignore[call-overload]
        kw_only=True,
    )
    """Experimental representation of the success value."""

    failure_value: ChoiceValue = field(
        default=False,
        validator=[instance_of(ChoiceValue), validate_not_nan],  # type: ignore[call-overload]
        kw_only=True,
    )
    """Experimental representation of the failure value."""

    @failure_value.validator
    def _validate_values(self, _, value):
        """Validate that the two choice values of the target are different."""
        if value == self.success_value:
            raise ValueError(
                f"The two choice values of a '{BinaryTarget.__name__}' must be "
                f"different but the following value was provided for both choices of "
                f"target '{self.name}': {value}"
            )

    @override
    def transform(
        self, series: pd.Series | None = None, /, *, data: pd.DataFrame | None = None
    ) -> pd.Series:
        # >>>>>>>>>> Deprecation
        if not ((series is None) ^ (data is None)):
            raise ValueError(
                "Provide the data to be transformed as first positional argument."
            )

        if data is not None:
            assert data.shape[1] == 1
            series = data.iloc[:, 0]
            warnings.warn(
                "Providing a dataframe via the `data` argument is deprecated and "
                "will be removed in a future version. Please pass your data "
                "in form of a series as positional argument instead.",
                DeprecationWarning,
            )

        # Mypy does not infer from the above that `series` must be a series here
        assert isinstance(series, pd.Series)
        # <<<<<<<<<< Deprecation

        # Validate target values
        invalid = series[
            ~series.isin([self.success_value, self.failure_value]).to_numpy()
        ]
        if len(invalid) > 0:
            raise InvalidTargetValueError(
                f"The following values entered for target '{self.name}' are not in the "
                f"set of accepted choice values "
                f"{set((self.success_value, self.failure_value))}: {set(invalid)}"
            )

        # Transform
        success_idx = series == self.success_value
        return pd.Series(
            np.where(success_idx, _SUCCESS_VALUE_COMP, _FAILURE_VALUE_COMP),
            index=series.index,
            name=series.name,
        )

    @override
    def summary(self) -> dict:
        return dict(
            Type=self.__class__.__name__,
            Name=self.name,
            Success_value=self.success_value,
            Failure_value=self.failure_value,
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
