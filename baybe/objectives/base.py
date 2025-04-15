"""Base classes for all objectives."""

import gc
from abc import ABC, abstractmethod
from typing import ClassVar

import cattrs
import pandas as pd
from attrs import define

from baybe.serialization.core import (
    converter,
    get_base_structure_hook,
    unstructure_base,
)
from baybe.serialization.mixin import SerialMixin
from baybe.targets.base import Target

# TODO: Reactive slots in all classes once cached_property is supported:
#   https://github.com/python-attrs/attrs/issues/164


@define(frozen=True, slots=False)
class Objective(ABC, SerialMixin):
    """Abstract base class for all objectives."""

    is_multi_output: ClassVar[bool]
    """Class variable indicating if the objective produces multiple outputs."""

    @property
    @abstractmethod
    def targets(self) -> tuple[Target, ...]:
        """The targets included in the objective."""

    @property
    @abstractmethod
    def n_outputs(self) -> int:
        """The number of outputs of the objective."""

    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
        /,
        *,
        allow_missing: bool = False,
        allow_extra: bool = False,
    ) -> pd.DataFrame:
        """Transform target values from experimental to computational representation.

        Args:
            df: The dataframe to be transformed. The allowed columns of the dataframe
                are dictated by the ``allow_missing`` and ``allow_extra`` flags.
            allow_missing: If ``False``, each target of the objective must have
                exactly one corresponding column in the given dataframe. If ``True``,
                the dataframe may contain only a subset of target columns.
            allow_extra: If ``False``, each column present in the dataframe must
                correspond to exactly one target of the objective. If ``True``, the
                dataframe may contain additional non-target-related columns, which
                will be ignored.

        Returns:
            A corresponding dataframe with the targets in computational representation.
        """


def to_objective(x: Target | Objective, /) -> Objective:
    """Convert a target into an objective (with objective passthrough)."""
    return x if isinstance(x, Objective) else x.to_objective()


# Register (un-)structure hooks
converter.register_structure_hook(
    Objective,
    get_base_structure_hook(
        Objective,
        overrides=dict(
            _target=cattrs.override(rename="target"),
            _targets=cattrs.override(rename="targets"),
        ),
    ),
)
converter.register_unstructure_hook(
    Objective,
    lambda x: unstructure_base(
        x,
        overrides=dict(
            _target=cattrs.override(rename="target"),
            _targets=cattrs.override(rename="targets"),
        ),
    ),
)
# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
