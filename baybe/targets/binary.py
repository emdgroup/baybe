"""Binary target."""

import pandas as pd
from attrs import define, field

from baybe.exceptions import UnknownTargetError
from baybe.serialization import SerialMixin
from baybe.targets.base import Target


@define(frozen=True)
class BinaryTarget(Target, SerialMixin):
    """Class for bernoulli targets."""

    positive_target = field(default=1)
    """Experimental representation of the positive target"""

    negative_target = field(default=0)
    """Experimental representation of the negative target"""

    # TODO: add optimization direction

    def transform_to_binary(self, experimental_representation):
        """Sample wise transform from experimental to computational representation."""
        if experimental_representation == self.positive_target:
            return 1
        elif experimental_representation == self.negative_target:
            return 0
        raise UnknownTargetError(
            f"The entered target '{experimental_representation}' is not in the"
            f" target set of {{{self.positive_target}, {self.negative_target}}}"
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
            Positive_target=self.positive_target,
            Negative_target=self.negative_target,
        )
