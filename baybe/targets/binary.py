"""Binary targets."""

from typing import TypeAlias

import pandas as pd
from attrs import define, field
from attrs.validators import instance_of

from baybe.exceptions import InvalidTargetValueError
from baybe.serialization import SerialMixin
from baybe.targets.base import Target

ChoiceValue: TypeAlias = bool | int | float | str
"""Types of values that a :class:`BinaryTarget` can take."""


@define(frozen=True)
class BinaryTarget(Target, SerialMixin):
    """Class for binary targets."""

    positive_value: ChoiceValue = field(
        default=1, validator=instance_of(ChoiceValue), kw_only=True
    )
    """Experimental representation of the positive value."""

    negative_value: ChoiceValue = field(
        default=0, validator=instance_of(ChoiceValue), kw_only=True
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
        data = data.copy()
        pos_idx = data.iloc[:, 0] == self.positive_value
        neg_idx = data.iloc[:, 0] == self.negative_value
        data[pos_idx] = 1
        data[neg_idx] = 0

        return data

    def summary(self) -> dict:  # noqa: D102
        # See base class.
        return dict(
            Type=self.__class__.__name__,
            Name=self.name,
            Positive_value=self.positive_value,
            Negative_value=self.negative_value,
        )
