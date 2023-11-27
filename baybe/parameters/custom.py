"""Custom parameters."""

from functools import cached_property
from typing import Any, ClassVar, Union

import pandas as pd
from attr import define, field

from baybe.parameters.base import DiscreteParameter
from baybe.parameters.enum import CustomEncoding
from baybe.parameters.validation import validate_decorrelation
from baybe.utils import df_uncorrelated_features, eq_dataframe


@define(frozen=True, slots=False)
class CustomDiscreteParameter(DiscreteParameter):
    """Custom parameters.

    For these parameters, the user can read in a precomputed representation for labels,
    e.g. from quantum chemistry.
    """

    # class variables
    is_numeric: ClassVar[bool] = False
    # See base class.

    # object variables
    data: pd.DataFrame = field(eq=eq_dataframe)
    """A mapping that provides the encoding for all available parameter values."""

    decorrelate: Union[bool, float] = field(
        default=True, validator=validate_decorrelation
    )
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
            ValueError: If the dataframe contains ``NaN``.
            ValueError: If the dataframe contains duplicated indices.
            ValueError: If the dataframe contains non-numeric values.
            ValueError: If the dataframe contains columns that only contain a single
                value.
        """
        if value.isna().any().any():
            raise ValueError(
                f"The custom dataframe for parameter {self.name} contains NaN "
                f"entries, which is not supported."
            )
        if len(value) != len(set(value.index)):
            raise ValueError(
                f"The custom dataframe for parameter {self.name} contains "
                f"duplicated indices. Please only provide dataframes with unique"
                f" indices."
            )
        if value.select_dtypes("number").shape[1] != value.shape[1]:
            raise ValueError(
                f"The custom dataframe for parameter {self.name} contains "
                f"non-numeric values."
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
        comp_df = self.data.rename(columns=lambda x: f"{self.name}_{x}")

        # Get a decorrelated subset of the provided features
        if self.decorrelate:
            if isinstance(self.decorrelate, bool):
                comp_df = df_uncorrelated_features(comp_df)
            else:
                comp_df = df_uncorrelated_features(comp_df, threshold=self.decorrelate)

        return comp_df
