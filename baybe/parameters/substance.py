"""Substance parameters."""

from functools import cached_property
from typing import Any, ClassVar, Dict, Union

import pandas as pd
from attr import define, field

from baybe.parameters.base import DiscreteParameter
from baybe.parameters.enum import SubstanceEncoding
from baybe.parameters.validation import validate_decorrelation
from baybe.utils import df_drop_single_value_columns, df_uncorrelated_features
from baybe.utils.chemistry import _MORDRED_INSTALLED, _RDKIT_INSTALLED

if _RDKIT_INSTALLED:
    from baybe.utils import (
        is_valid_smiles,
        smiles_to_fp_features,
        smiles_to_rdkit_features,
    )

    if _MORDRED_INSTALLED:
        from baybe.utils import smiles_to_mordred_features


@define(frozen=True, slots=False)
class SubstanceParameter(DiscreteParameter):
    """Generic substances that are treated with cheminformatics descriptors.

    Only a decorrelated subset of descriptors should be used as otherwise this can
    result in a large number of features. For a handful of molecules, keeping only
    descriptors that have a maximum correlation of 0.7 reduces the number of
    descriptors to about 5-20. The number might be substantially higher with more
    labels given.
    """

    # class variables
    is_numeric: ClassVar[bool] = False
    # See base class.

    # object variables
    data: Dict[str, str] = field()
    """A mapping that provides the SMILES strings for all available parameter values."""

    decorrelate: Union[bool, float] = field(
        default=True, validator=validate_decorrelation
    )
    """Specifies the used decorrelation mode for the parameter encoding.

        - ``False``: The encoding is used as is.
        - ``True``: The encoding is decorrelated using a default correlation threshold.
        - float in (0, 1): The encoding is decorrelated using the specified threshold.
    """

    encoding: SubstanceEncoding = field(
        default=SubstanceEncoding.MORDRED, converter=SubstanceEncoding
    )
    # See base class.

    @encoding.validator
    def _validate_encoding(self, _: Any, value: str) -> None:  # noqa: DOC101, DOC103
        """Validate that the chosen encoding can be used.

        This validation is necessary since certain encodings are only usable when
        additional dependencies, in particular the ``chem`` dependency, have been
        installed.

        Raises:
            ImportError: If the ``chem``dependency was not installed but an encoding
                requiring this dependency is requested.
        """
        if value is SubstanceEncoding.MORDRED and not (
            _MORDRED_INSTALLED and _RDKIT_INSTALLED
        ):
            raise ImportError(
                "The mordred/rdkit packages are not installed, a SubstanceParameter "
                "with MORDRED encoding cannot be used. Consider installing baybe with "
                "'chem' dependency like 'pip install baybe[chem]'"
            )
        if (
            value in [SubstanceEncoding.RDKIT, SubstanceEncoding.MORGAN_FP]
            and not _RDKIT_INSTALLED
        ):
            raise ImportError(
                "The rdkit package is not installed, a SubstanceParameter with "
                "RDKIT or MORGAN_FP encoding cannot be used. Consider installing baybe "
                "with 'chem' dependency like 'pip install baybe[chem]'"
            )

    @data.validator
    def _validate_substance_data(  # noqa: DOC101, DOC103
        self, _: Any, value: Dict[str, str]
    ) -> None:
        """Validate that the substance data, provided as SMILES, is valid.

        Raises:
            ValueError: If one or more of the SMILES are invalid.
        """
        for name, smiles in value.items():
            if _RDKIT_INSTALLED and not is_valid_smiles(smiles):
                raise ValueError(
                    f"The SMILES '{smiles}' for molecule '{name}' does "
                    f"not appear to be valid."
                )

    @property
    def values(self) -> tuple:
        """Returns the labels of the given set of molecules."""
        # Since the order of dictionary keys is important here, this will only work
        # for Python 3.7 or higher
        return tuple(self.data.keys())

    @cached_property
    def comp_df(self) -> pd.DataFrame:  # noqa: D102
        # See base class.
        vals = list(self.data.values())
        pref = self.name + "_"

        # Get the raw descriptors
        if self.encoding is SubstanceEncoding.MORDRED:
            comp_df = smiles_to_mordred_features(vals, prefix=pref)
        elif self.encoding is SubstanceEncoding.RDKIT:
            comp_df = smiles_to_rdkit_features(vals, prefix=pref)
        elif self.encoding is SubstanceEncoding.MORGAN_FP:
            comp_df = smiles_to_fp_features(vals, prefix=pref)
        else:
            raise ValueError(
                f"Unknown parameter encoding {self.encoding} for parameter {self.name}."
            )

        # Drop NaN and constant columns
        comp_df = comp_df.loc[:, ~comp_df.isna().any(axis=0)]
        comp_df = df_drop_single_value_columns(comp_df)

        # If there are bool columns, convert them to int (possible for Mordred)
        comp_df.loc[:, comp_df.dtypes == bool] = comp_df.loc[
            :, comp_df.dtypes == bool
        ].astype(int)

        # Label the rows with the molecule names
        comp_df.index = pd.Index(self.values)

        # Get a decorrelated subset of the descriptors
        if self.decorrelate:
            if isinstance(self.decorrelate, bool):
                comp_df = df_uncorrelated_features(comp_df)
            else:
                comp_df = df_uncorrelated_features(comp_df, threshold=self.decorrelate)

        return comp_df
