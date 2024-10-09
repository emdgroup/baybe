"""Base functionality for all BayBE targets."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd
from attrs import define, field

from baybe.serialization import (
    SerialMixin,
    converter,
    get_base_structure_hook,
    unstructure_base,
)

if TYPE_CHECKING:
    from baybe.objective import SingleTargetObjective


@define(frozen=True)
class Target(ABC, SerialMixin):
    """Abstract base class for all target variables.

    Stores information about the range, transformations, etc.
    """

    name: str = field()
    """The name of the target."""

    def to_objective(self) -> SingleTargetObjective:
        """Create a single-task objective from the target."""
        from baybe.objectives.single import SingleTargetObjective

        return SingleTargetObjective(self)

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

    @abstractmethod
    def summary(self) -> dict:
        """Return a custom summarization of the target."""

    def __str__(self) -> str:
        return str(self.summary())


# Register (un-)structure hooks
converter.register_structure_hook(Target, get_base_structure_hook(Target))
converter.register_unstructure_hook(Target, unstructure_base)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
