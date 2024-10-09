"""Custom parameters."""

import gc
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from attrs import define, field
from attrs.validators import min_len

from baybe.parameters.base import DiscreteParameter
from baybe.parameters.enum import CustomEncoding
from baybe.parameters.validation import validate_decorrelation
from baybe.utils.boolean import eq_dataframe
from baybe.utils.dataframe import df_uncorrelated_features
from baybe.utils.numerical import DTypeFloatNumpy


@define(frozen=True, slots=False)
class CustomDiscreteParameter(DiscreteParameter):
    """Custom parameters.

    For these parameters, the user can read in a precomputed representation for labels,
    e.g. from quantum chemistry.
    """

    # class variables
    is_numerical: ClassVar[bool] = False
    # See base class.

    # object variables
    data: pd.DataFrame = field(validator=min_len(2), eq=eq_dataframe)
    """A mapping that provides the encoding for all available parameter values."""

    decorrelate: bool | float = field(default=True, validator=validate_decorrelation)
    """Specifies the used decorrelation mode for the parameter encoding.

        - ``False``: The encoding is used as is.
        - ``True``: The encoding is decorrelated using a default correlation threshold.
        - float in (0, 1): The encoding is decorrelated using the specified threshold.
    """

    encoding: CustomEncoding = field(init=False, default=CustomEncoding.CUSTOM)
    # See base class.

    @data.validator
    def _validate_custom_data(  # noqa: DOC101, DOC103
        self, _: Any, value: pd.DataFrame
    ) -> None:
        """Validate the dataframe with the custom representation.

        Raises:
            ValueError: If the dataframe contains non-numeric values.
            ValueError: If the dataframe index contains non-string values.
            ValueError: If the dataframe index contains empty strings.
            ValueError: If the dataframe contains ``NaN``.
            ValueError: If the dataframe index contains duplicates.
            ValueError: If the dataframe contains columns with only one unique value.
        """
        if value.select_dtypes("number").shape[1] != value.shape[1]:
            raise ValueError(
                f"The custom dataframe for parameter {self.name} contains "
                f"non-numeric values."
            )
        if not all(isinstance(x, str) for x in value.index):
            raise ValueError(
                f"The custom dataframe for parameter {self.name} contains non-string "
                f"index values."
            )
        if not all(len(x) > 0 for x in value.index):
            raise ValueError(
                f"The custom dataframe for parameter {self.name} contains empty string "
                f"index values."
            )
        if not np.isfinite(value.values).all():
            raise ValueError(
                f"The custom dataframe for parameter {self.name} contains nan/infinity "
                f"entries."
            )
        if len(value) != len(set(value.index)):
            raise ValueError(
                f"The custom dataframe for parameter {self.name} contains "
                f"duplicated indices."
            )
        if any(value.nunique() == 1):
            raise ValueError(
                f"The custom dataframe for parameter {self.name} has columns "
                "that contain only a single value and hence carry no information."
            )

    @property
    def values(self) -> tuple:
        """Returns the representing labels of the parameter."""
        return tuple(self.data.index)

    @cached_property
    def comp_df(self) -> pd.DataFrame:  # noqa: D102
        # See base class.
        # The encoding is directly provided by the user
        # We prepend the parameter name to the columns names to avoid potential
        # conflicts with other parameters
        comp_df = self.data.rename(columns=lambda x: f"{self.name}_{x}").astype(
            DTypeFloatNumpy
        )

        # Get a decorrelated subset of the provided features
        if self.decorrelate:
            if isinstance(self.decorrelate, bool):
                comp_df = df_uncorrelated_features(comp_df)
            else:
                comp_df = df_uncorrelated_features(comp_df, threshold=self.decorrelate)

        return comp_df


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
