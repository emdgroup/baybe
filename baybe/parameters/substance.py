"""Substance parameters."""

import gc
from functools import cached_property
from typing import Any, ClassVar

import pandas as pd
from attrs import define, field
from attrs.validators import and_, deep_mapping, instance_of, min_len

from baybe.parameters.base import DiscreteParameter
from baybe.parameters.enum import SubstanceEncoding
from baybe.parameters.validation import validate_decorrelation
from baybe.utils.basic import group_duplicate_values
from baybe.utils.dataframe import df_drop_single_value_columns, df_uncorrelated_features

try:  # For python < 3.11, use the exceptiongroup backport
    ExceptionGroup
except NameError:
    from exceptiongroup import ExceptionGroup


Smiles = str
"""Type alias for SMILES strings."""


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
    is_numerical: ClassVar[bool] = False
    # See base class.

    # object variables
    data: dict[str, Smiles] = field(
        converter=lambda x: dict(sorted(x.items())),
        validator=deep_mapping(
            mapping_validator=min_len(2),
            # FIXME[typing]: https://github.com/python-attrs/attrs/issues/1206
            key_validator=and_(instance_of(str), min_len(1)),
            value_validator=lambda *x: None,
        ),
    )
    """A mapping that provides the SMILES strings for all available parameter values."""

    decorrelate: bool | float = field(default=True, validator=validate_decorrelation)
    """Specifies the used decorrelation mode for the parameter encoding.

        - ``False``: The encoding is used as is.
        - ``True``: The encoding is decorrelated using a default correlation threshold.
        - float in (0, 1): The encoding is decorrelated using the specified threshold.
    """

    encoding: SubstanceEncoding = field(
        default=SubstanceEncoding.MORDRED, converter=SubstanceEncoding
    )
    # See base class.

    @data.validator
    def _validate_substance_data(  # noqa: DOC101, DOC103
        self, _: Any, data: dict[str, Smiles]
    ) -> None:
        """Validate that the substance data, provided as SMILES, is valid.

        Raises:
            ValueError: If one or more of the SMILES are invalid.
            ValueError: If the several entries represent the same substance.
        """
        from baybe.utils import chemistry

        # Check for invalid SMILES
        canonical_smiles = {}
        exceptions = []
        for name, smiles in data.items():
            try:
                canonical_smiles[name] = chemistry.get_canonical_smiles(smiles)
            except ValueError:
                exceptions.append(
                    ValueError(
                        f"The SMILES '{smiles}' for molecule '{name}' does "
                        f"not appear to be valid."
                    )
                )
        if exceptions:
            raise ExceptionGroup("invalid SMILES", exceptions)

        # Check for duplicate substances
        if groups := group_duplicate_values(canonical_smiles):
            exceptions = []
            for group, substances in groups.items():
                group_data = {s: data[s] for s in substances}
                exceptions.append(
                    ValueError(
                        f"The following entries all represent the same substance "
                        f"'{group}': {group_data}."
                    )
                )
            raise ExceptionGroup("duplicate substances", exceptions)

    @property
    def values(self) -> tuple:
        """Returns the labels of the given set of molecules."""
        return tuple(self.data.keys())

    @cached_property
    def comp_df(self) -> pd.DataFrame:  # noqa: D102
        # See base class.

        from baybe.utils import chemistry

        vals = list(self.data.values())
        pref = self.name + "_"

        # Get the raw descriptors
        if self.encoding is SubstanceEncoding.MORDRED:
            comp_df = chemistry.smiles_to_mordred_features(vals, prefix=pref)
        elif self.encoding is SubstanceEncoding.RDKIT:
            comp_df = chemistry.smiles_to_rdkit_features(vals, prefix=pref)
        elif self.encoding is SubstanceEncoding.MORGAN_FP:
            comp_df = chemistry.smiles_to_fp_features(vals, prefix=pref)
        else:
            raise ValueError(
                f"Unknown parameter encoding {self.encoding} for parameter {self.name}."
            )

        # Drop NaN and constant columns
        comp_df = comp_df.loc[:, ~comp_df.isna().any(axis=0)]
        comp_df = df_drop_single_value_columns(comp_df)

        # If there are bool columns, convert them to int (possible for Mordred)
        bool_cols = comp_df.select_dtypes(bool).columns
        comp_df[bool_cols] = comp_df[bool_cols].astype(int)

        # Label the rows with the molecule names
        comp_df.index = pd.Index(self.values)

        # Get a decorrelated subset of the descriptors
        if self.decorrelate:
            if isinstance(self.decorrelate, bool):
                comp_df = df_uncorrelated_features(comp_df)
            else:
                comp_df = df_uncorrelated_features(comp_df, threshold=self.decorrelate)

        return comp_df


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
