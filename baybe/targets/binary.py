"""Binary target."""

from typing import TypeAlias

import pandas as pd
from attrs import define, field
from attrs.validators import instance_of

from baybe.exceptions import UnknownTargetError
from baybe.serialization import SerialMixin
from baybe.targets.base import Target

ChoiceValue: TypeAlias = bool | int | float | str
"""Types of values that a :class:`BinaryTarget` can take."""


@define(frozen=True)
class BinaryTarget(Target, SerialMixin):
    """Class for bernoulli targets."""

    positive_value: ChoiceValue = field(
        default=1, validator=instance_of(ChoiceValue), kw_only=True
    )
    """Experimental representation of the positive target"""

    negative_value: ChoiceValue = field(
        default=0, validator=instance_of(ChoiceValue), kw_only=True
    )
    """Experimental representation of the negative target"""

    @negative_value.validator
    def _validate_values(self, _, value):
        """Validate that the two choice values of the target are different."""
        if value == self.positive_value:
            raise ValueError(
                f"The two choice values of a '{BinaryTarget.__name__}' must be "
                f"different but the following value was provided for both choices of "
                f"target '{self.name}': {value}"
            )

    def transform_to_binary(self, experimental_representation):
        """Sample wise transform from experimental to computational representation."""
        if experimental_representation == self.positive_value:
            return 1
        elif experimental_representation == self.negative_value:
            return 0
        raise UnknownTargetError(
            f"The entered target '{experimental_representation}' is not in the"
            f" target set of {{{self.positive_value}, {self.negative_value}}}"
        )

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        # see base class
        assert data.shape[1] == 1
        data[data.columns[0]] = data[data.columns[0]].apply(self.transform_to_binary)
        return data

    def summary(self) -> dict:  # noqa: D102
        # see base class
        return dict(
            Type=self.__class__.__name__,
            Name=self.name,
            Positive_value=self.positive_value,
            Negative_value=self.negative_value,
        )
