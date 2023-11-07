"""Base functionality for all BayBE targets."""

from abc import ABC, abstractmethod

import pandas as pd
from attr import define, field


@define(frozen=True)
class Target(ABC):
    """Abstract base class for all target variables.

    Stores information about the range, transformations, etc.
    """

    name: str = field()
    """The name of the target."""

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data into computational representation.

        The transformation depends on the target mode, e.g. minimization, maximization,
        matching, etc.

        Args:
            data: The data to be transformed.

        Returns:
            A dataframe containing the transformed data.
        """
