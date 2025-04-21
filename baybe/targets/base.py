"""Base functionality for all BayBE targets."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd
from attrs import define, field
from typing_extensions import override

from baybe.serialization import (
    SerialMixin,
    converter,
    get_base_structure_hook,
    unstructure_base,
)

if TYPE_CHECKING:
    from baybe.objectives import SingleTargetObjective


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
    def transform(self, series: pd.Series, /) -> pd.Series:
        """Transform target measurements to computational representation.

        Args:
            series: The target measurements in experimental representation to be
                transformed.

        Returns:
            A series containing the transformed measurements. The series name matches
            that of the input.
        """

    @abstractmethod
    def summary(self) -> dict:
        """Return a custom summarization of the target."""

    @override
    def __str__(self) -> str:
        return str(self.summary())


# Register (un-)structure hooks
converter.register_structure_hook(Target, get_base_structure_hook(Target))
converter.register_unstructure_hook(Target, unstructure_base)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
