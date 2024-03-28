"""Base classes for all objectives."""

from abc import ABC, abstractmethod

import pandas as pd
from attrs import define

from baybe.objectives.deprecation import structure_objective
from baybe.serialization.core import (
    converter,
    unstructure_base,
)
from baybe.serialization.mixin import SerialMixin
from baybe.targets.base import Target


@define(frozen=True)
class Objective(ABC, SerialMixin):
    """Abstract base class for all objectives."""

    @property
    @abstractmethod
    def targets(self) -> tuple[Target, ...]:
        """The targets included in the objective."""

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform targets from experimental to computational representation.

        Args:
            data: The data to be transformed. Must contain columns for all targets
                but can contain additional columns.

        Returns:
            A new dataframe with the targets in computational representation.
        """


converter.register_structure_hook(Objective, structure_objective)
converter.register_unstructure_hook(Objective, unstructure_base)
