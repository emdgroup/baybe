"""Base class for symmetries."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

import pandas as pd
from attrs import define, field
from attrs.validators import instance_of

from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.serialization import SerialMixin

if TYPE_CHECKING:
    from baybe.parameters.base import Parameter
    from baybe.searchspace import SearchSpace


@define(frozen=True)
class Symmetry(SerialMixin, ABC):
    """Abstract base class for symmetries.

    A ``Symmetry`` is a concept that can be used to configure the modeling process in
    the presence of invariances.
    """

    use_data_augmentation: bool = field(
        default=True, validator=instance_of(bool), kw_only=True
    )
    """Flag indicating whether data augmentation is to be used."""

    @property
    @abstractmethod
    def parameter_names(self) -> tuple[str, ...]:
        """The names of the parameters affected by the symmetry."""

    def summary(self) -> dict:
        """Return a custom summarization of the symmetry."""
        symmetry_dict = dict(
            Type=self.__class__.__name__, Affected_Parameters=self.parameter_names
        )
        return symmetry_dict

    @abstractmethod
    def augment_measurements(
        self,
        measurements: pd.DataFrame,
        parameters: Iterable[Parameter] | None = None,
    ) -> pd.DataFrame:
        """Augment the given measurements according to the symmetry.

        Args:
            measurements: The dataframe containing the measurements to be
                augmented.
            parameters: Optional parameter objects carrying additional information.
                Only required by specific augmentation implementations.

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
        parameters_missing = set(self.parameter_names).difference(
            searchspace.parameter_names
        )
        if parameters_missing:
            raise IncompatibleSearchSpaceError(
                f"The symmetry of type '{self.__class__.__name__}' was set up with the "
                f"following parameters that are not present in the search space: "
                f"{parameters_missing}."
            )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
