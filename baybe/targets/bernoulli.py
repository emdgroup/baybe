"""Bernoulli target."""

import pandas as pd
from attrs import define

from baybe.serialization import SerialMixin
from baybe.targets.base import Target


@define(frozen=True)
class BernoulliTarget(Target, SerialMixin):
    """Class for bernoulli targets."""

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        # see base class
        assert data.shape[1] == 1
        # TODO: negation (1 - data) for min mode?!
        return data

    def summary(self) -> dict:  # noqa: D102
        # see base class
        return dict(
            Type=self.__class__.__name__,
            Name=self.name,
        )
