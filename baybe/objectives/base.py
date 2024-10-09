"""Base classes for all objectives."""

import gc
from abc import ABC, abstractmethod

import pandas as pd
from attrs import define
from cattrs import override

from baybe.objectives.deprecation import structure_objective
from baybe.serialization.core import (
    converter,
    unstructure_base,
)
from baybe.serialization.mixin import SerialMixin
from baybe.targets.base import Target

# TODO: Reactive slots in all classes once cached_property is supported:
#   https://github.com/python-attrs/attrs/issues/164


@define(frozen=True, slots=False)
class Objective(ABC, SerialMixin):
    """Abstract base class for all objectives."""

    @property
    @abstractmethod
    def targets(self) -> tuple[Target, ...]:
        """The targets included in the objective."""

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform target values from experimental to computational representation.

        Args:
            data: The data to be transformed. Must contain columns for all targets
                but can contain additional columns.

        Returns:
            A new dataframe with the targets in computational representation.
        """


def to_objective(x: Target | Objective, /) -> Objective:
    """Convert a target into an objective (with objective passthrough)."""
    return x if isinstance(x, Objective) else x.to_objective()


# Register de-/serialization hooks
converter.register_structure_hook(Objective, structure_objective)
converter.register_unstructure_hook(
    Objective,
    lambda x: unstructure_base(
        x,
        overrides=dict(
            _target=override(rename="target"),
            _targets=override(rename="targets"),
        ),
    ),
)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
