"""Base functionality for all BayBE targets."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd
from attrs import define, field
from attrs.validators import instance_of
from typing_extensions import override

from baybe.serialization import (
    SerialMixin,
)
from baybe.utils.metadata import MeasurableMetadata, to_metadata

if TYPE_CHECKING:
    from baybe.objectives import SingleTargetObjective


@define(frozen=True)
class Target(ABC, SerialMixin):
    """Abstract base class for all target variables.

    Stores information about the range, transformations, etc.
    """

    name: str = field(validator=instance_of(str))
    """The name of the target."""

    metadata: MeasurableMetadata = field(
        factory=MeasurableMetadata,
        converter=lambda x: to_metadata(x, MeasurableMetadata),
        kw_only=True,
    )
    """Optional metadata containing description, unit, and other information."""

    @property
    def description(self) -> str | None:
        """The description of the target."""
        return self.metadata.description

    @property
    def unit(self) -> str | None:
        """The unit of measurement for the target."""
        return self.metadata.unit

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


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
