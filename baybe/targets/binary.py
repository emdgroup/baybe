"""Binary targets."""

from __future__ import annotations

import gc
import warnings
from typing import TYPE_CHECKING, TypeAlias

import narwhals.stable.v2 as nw
import pandas as pd
from attrs import define, field
from attrs.validators import instance_of
from typing_extensions import override

from baybe.exceptions import InvalidTargetValueError
from baybe.serialization import SerialMixin
from baybe.targets.base import Target
from baybe.utils.validation import validate_not_nan

if TYPE_CHECKING:
    from narwhals.typing import IntoSeriesT

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
        validator=[instance_of(ChoiceValue), validate_not_nan],
        kw_only=True,
    )
    """Experimental representation of the success value."""

    failure_value: ChoiceValue = field(
        default=False,
        validator=[instance_of(ChoiceValue), validate_not_nan],
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
        self, series: IntoSeriesT | None = None, /, *, data: pd.DataFrame | None = None
    ) -> IntoSeriesT:
        # >>>>>>>>>> Deprecation
        if not ((series is None) ^ (data is None)):
            raise ValueError(
                "Provide the data to be transformed as first positional argument."
            )

        if data is not None:
            assert data.shape[1] == 1
            series = data.iloc[:, 0]  # type: ignore[assignment]
            warnings.warn(
                "Providing a dataframe via the `data` argument is deprecated and "
                "will be removed in a future version. Please pass your data "
                "in form of a series as positional argument instead.",
                DeprecationWarning,
            )

        assert series is not None
        # <<<<<<<<<< Deprecation

        nw_series = nw.from_native(series, series_only=True)
        choices = [self.success_value, self.failure_value]

        # Validate target values
        invalid = nw_series.filter(~nw_series.is_in(choices))
        if len(invalid) > 0:
            raise InvalidTargetValueError(
                f"The following values entered for target '{self.name}' are not in the "
                f"set of accepted choice values "
                f"{set(choices)}: {set(invalid.to_list())}"
            )

        # Transform
        return (
            nw_series.to_frame()
            .select(
                nw.when(nw.col(nw_series.name) == self.success_value)
                .then(_SUCCESS_VALUE_COMP)
                .otherwise(_FAILURE_VALUE_COMP)
                .alias(nw_series.name)
            )[nw_series.name]
            .to_native()
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
