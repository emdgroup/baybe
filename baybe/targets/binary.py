"""Binary targets."""

from typing import TypeAlias

import numpy as np
import pandas as pd
from attrs import define, field
from attrs.validators import instance_of

from baybe.exceptions import InvalidTargetValueError
from baybe.serialization import SerialMixin
from baybe.targets.base import Target

ChoiceValue: TypeAlias = bool | int | float | str
"""Types of values that a :class:`BinaryTarget` can take."""

_POSITIVE_VALUE_COMP = True
"""Computational representation of the positive choice value."""

_NEGATIVE_VALUE_COMP = False
"""Computational representation of the negative choice value."""


@define(frozen=True)
class BinaryTarget(Target, SerialMixin):
    """Class for binary targets."""

    # FIXME[typing]: https://github.com/python-attrs/attrs/issues/1336

    positive_value: ChoiceValue = field(
        default=1,
        validator=instance_of(ChoiceValue),  # type: ignore[call-overload]
        kw_only=True,
    )
    """Experimental representation of the positive value."""

    negative_value: ChoiceValue = field(
        default=0,
        validator=instance_of(ChoiceValue),  # type: ignore[call-overload]
        kw_only=True,
    )
    """Experimental representation of the negative value."""

    @negative_value.validator
    def _validate_values(self, _, value):
        """Validate that the two choice values of the target are different."""
        if value == self.positive_value:
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
        invalid = col[~col.isin([self.positive_value, self.negative_value]).values]
        if len(invalid) > 0:
            raise InvalidTargetValueError(
                f"The following values entered for target '{self.name}' are not in the "
                f"set of accepted choice values "
                f"{set([self.positive_value, self.negative_value])}: \n{invalid}"
            )

        # Transform
        pos_idx = data.iloc[:, 0] == self.positive_value
        neg_idx = data.iloc[:, 0] == self.negative_value
        data = pd.DataFrame(
            np.zeros_like(data.values, dtype=bool),
            index=data.index,
            columns=data.columns,
        )
        data[pos_idx] = _POSITIVE_VALUE_COMP
        data[neg_idx] = _NEGATIVE_VALUE_COMP

        return data

    def summary(self) -> dict:  # noqa: D102
        # See base class.
        return dict(
            Type=self.__class__.__name__,
            Name=self.name,
            Positive_value=self.positive_value,
            Negative_value=self.negative_value,
        )
