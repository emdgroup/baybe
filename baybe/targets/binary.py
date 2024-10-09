"""Binary targets."""

import gc
from typing import TypeAlias

import numpy as np
import pandas as pd
from attrs import define, field
from attrs.validators import instance_of

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

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        # TODO: The method (signature) needs to be refactored, potentially when
        #   enabling multi-target settings. The current input type suggests that passing
        #   dataframes is allowed, but the code was designed for single targets and
        #   desirability objectives, where only one column is present.
        assert data.shape[1] == 1

        # Validate target values
        col = data.iloc[:, [0]]
        invalid = col[~col.isin([self.success_value, self.failure_value]).values]
        if len(invalid) > 0:
            raise InvalidTargetValueError(
                f"The following values entered for target '{self.name}' are not in the "
                f"set of accepted choice values "
                f"{{self.success_value, self.failure_value}}: \n{invalid}"
            )

        # Transform
        success_idx = data.iloc[:, 0] == self.success_value
        return pd.DataFrame(
            np.where(success_idx, _SUCCESS_VALUE_COMP, _FAILURE_VALUE_COMP),
            index=data.index,
            columns=data.columns,
        )

    def summary(self) -> dict:  # noqa: D102
        # See base class.
        return dict(
            Type=self.__class__.__name__,
            Name=self.name,
            Success_value=self.success_value,
            Failure_value=self.failure_value,
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
