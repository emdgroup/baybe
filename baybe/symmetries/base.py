"""Base class for symmetries."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd
from attrs import define

from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.serialization import SerialMixin

if TYPE_CHECKING:
    from baybe.searchspace import SearchSpace


@define(frozen=True)
class Symmetry(SerialMixin, ABC):
    """Abstract base class for symmetries.

    A ``Symmetry`` is a concept that can be used to configure the modeling process in
    the presence of invariances.
    """

    @property
    @abstractmethod
    def parameter_names(self) -> tuple[str, ...]:
        """The names of the parameters affected by the symmetry."""

    def summary(self) -> dict:
        """Return a custom summarization of the symmetry."""
        return dict(
            Type=self.__class__.__name__,
            Affected_Parameters=self.parameter_names,
        )

    def augment_measurements(
        self,
        measurements: pd.DataFrame,
        searchspace: SearchSpace,
    ) -> pd.DataFrame:
        """Augment the given measurements according to the symmetry.

        Args:
            measurements: The dataframe containing the measurements to be augmented.
            searchspace: The searchspace providing parameter context for augmentation.

        Returns:
            The augmented dataframe including the original measurements.
        """
        self.validate_searchspace_context(searchspace)
        return self._augment_measurements(measurements, searchspace)

    @abstractmethod
    def _augment_measurements(
        self,
        measurements: pd.DataFrame,
        searchspace: SearchSpace,
    ) -> pd.DataFrame:
        """Augment measurements (core logic for subclasses).

        Args:
            measurements: The dataframe containing the measurements to be augmented.
            searchspace: The searchspace providing parameter context for augmentation.

        Returns:
            The augmented dataframe including the original measurements.
        """

    def validate_searchspace_context(self, searchspace: SearchSpace) -> None:
        """Validate that the symmetry is compatible with the given searchspace.

        Args:
            searchspace: The searchspace to validate against.

        Raises:
            IncompatibleSearchSpaceError: If the symmetry affects parameters not
                present in the searchspace.
        """
        if missing := set(self.parameter_names) - set(searchspace.parameter_names):
            raise IncompatibleSearchSpaceError(
                f"The symmetry of type '{self.__class__.__name__}' was set up with the "
                f"following parameters that are not present in the search space: "
                f"{missing}."
            )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
