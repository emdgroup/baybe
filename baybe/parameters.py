"""
Functionality for different experimental parameter types.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import cached_property
from typing import (
    Any,
    ClassVar,
    Dict,
    get_args,
    get_type_hints,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from pydantic import confloat, Extra, StrictBool, validator
from pydantic.types import conlist
from sklearn.metrics.pairwise import pairwise_distances

from baybe.utils import ABCBaseModel
from .utils import (
    df_drop_single_value_columns,
    df_drop_string_columns,
    df_uncorrelated_features,
    is_valid_smiles,
    smiles_to_fp_features,
    smiles_to_mordred_features,
    smiles_to_rdkit_features,
    StrictValidationError,
)

log = logging.getLogger(__name__)


def _validate_value_list(lst: list, values: dict):
    """A pydantic validator to verify parameter values."""
    if len(lst) < 2:
        raise ValueError(
            f"Parameter {values['name']} must have at least two unique values."
        )
    if len(lst) != len(np.unique(lst)):
        raise ValueError(
            f"Values for parameter {values['name']} are not unique. "
            f"This would cause duplicates in the set of possible experiments."
        )
    return lst


class Parameter(ABC, ABCBaseModel):
    """
    Abstract base class for all parameters. Stores information about the
    type, range, constraints, etc. and handles in-range checks, transformations etc.
    """

    # class variables
    encoding: ClassVar[Optional[str]]
    is_numeric: ClassVar[bool] = False  # default that is changed for numeric parameters
    is_discrete: ClassVar[bool]

    # object variables
    name: str

    class Config:  # pylint: disable=missing-class-docstring
        extra = Extra.forbid
        keep_untouched = (
            cached_property,
        )  # required due to: https://github.com/pydantic/pydantic/issues/1241
        arbitrary_types_allowed = True
        json_encoders = {
            pd.DataFrame: lambda x: x.to_dict(orient="list"),
        }

    def is_in_range(self, item: object) -> bool:
        """
        Tells whether an item is within the parameter range.
        """
        # TODO: in terms of coding style, this is not ideal: `values` is currently only
        #  defined in the subclasses but not in the base class since it is either a
        #  member or a property, depending on the parameter type --> better solution?
        return item in self.values


class DiscreteParameter(Parameter, ABC):
    """
    Abstract class for discrete parameters.
    """

    # class variables
    is_discrete = True

    @cached_property
    @abstractmethod
    def comp_df(self) -> pd.DataFrame:
        """
        Returns the computational representation of the parameter.
        """

    def transform_rep_exp2comp(self, data: pd.Series = None) -> pd.DataFrame:
        """
        Transforms data from experimental to computational representation.

        Parameters
        ----------
        data: pd.Series
            Data to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed version of the data.
        """
        if self.encoding:
            # replace each label with the corresponding encoding
            transformed = pd.merge(
                left=data.rename("Labels").to_frame(),
                left_on="Labels",
                right=self.comp_df,
                right_index=True,
                how="left",
            ).drop(columns="Labels")
        else:
            transformed = data.to_frame()

        return transformed


class Categorical(DiscreteParameter):
    """
    Parameter class for categorical parameters.
    """

    # object variables
    values: conlist(Any, unique_items=True)
    encoding: Literal["OHE", "INT"] = "OHE"

    # validators
    _validated_values = validator("values", allow_reuse=True)(_validate_value_list)

    @cached_property
    def comp_df(self) -> pd.DataFrame:
        """
        See base class.
        """
        if self.encoding == "OHE":
            cols = [f"{self.name}_{val}" for val in self.values]
            comp_df = pd.DataFrame(np.eye(len(self.values), dtype=int), columns=cols)
        elif self.encoding == "INT":
            comp_df = pd.DataFrame(range(len(self.values)), columns=[self.name])
        comp_df.index = self.values

        return comp_df


class NumericDiscrete(DiscreteParameter):
    """
    Parameter class for discrete numerical parameters (a.k.a. setpoints).
    """

    # class variables
    is_numeric = True
    encoding = None

    # object variables
    values: conlist(float, unique_items=True)
    tolerance: float = 0.0

    # validators
    _validated_values = validator("values", allow_reuse=True)(_validate_value_list)

    @validator("tolerance")
    def validate_tolerance(cls, tolerance, values):
        """
        Validates that the tolerance (i.e. allowed experimental uncertainty when
        reading in measured values) is safe. A tolerance larger than half the minimum
        distance between parameter values is not allowed because that could cause
        ambiguity when inputting data points later.
        """
        # NOTE: computing all pairwise distances can be avoided if we ensure that the
        #   values are ordered (which is currently not the case)
        dists = pairwise_distances(np.asarray(values["values"]).reshape(-1, 1))
        np.fill_diagonal(dists, np.inf)
        max_tol = dists.min() / 2.0

        if tolerance >= max_tol:
            raise ValueError(
                f"Parameter {values['name']} is initialized with tolerance "
                f"{tolerance} but due to the values {values['values']} a "
                f"maximum tolerance of {max_tol} is suggested to avoid ambiguity."
            )

        return tolerance

    @cached_property
    def comp_df(self) -> pd.DataFrame:
        """
        See base class.
        """
        comp_df = pd.DataFrame({self.name: self.values}, index=self.values)
        return comp_df

    def is_in_range(self, item: float) -> bool:
        """
        See base class.
        """
        differences_acceptable = [
            np.abs(val - item) <= self.tolerance for val in self.values
        ]
        return any(differences_acceptable)


class NumericContinuous(Parameter):
    """
    Parameter class for continuous numerical parameters.
    """

    # class variables
    is_numeric = True
    is_discrete = False

    # object variables
    bounds: Tuple[Optional[float], Optional[float]]

    @validator("bounds")
    def validate_bounds(cls, bounds):
        """
        Validate boundaries
        """
        bounds = list(bounds)
        if bounds[0] is None:
            bounds[0] = -np.inf

        if bounds[1] is None:
            bounds[1] = np.inf

        if bounds[1] <= bounds[0]:
            raise StrictValidationError(
                "Bounds for continuous parameters must be unique and in ascending "
                "order. They may contain -np.nan/np.nan or None in case there is no "
                "bound."
            )

        return tuple(bounds)

    def is_in_range(self, item: float) -> bool:
        """
        See base class.
        """

        return self.bounds[0] <= item <= self.bounds[1]


class GenericSubstance(DiscreteParameter):
    """
    Parameter class for generic substances that are treated with cheminformatics
    descriptors.

    Only a decorrelated subset of descriptors should be used as otherwise this can
    result in a large number of features. For a handful of molecules, keeping only
    descriptors that have a maximum correlation of 0.7 reduces the number of
    descriptors to about 5-20. The number might be substantially higher with more
    labels given.
    """

    # object variables
    decorrelate: Union[StrictBool, confloat(gt=0.0, lt=1.0, strict=True)] = True
    encoding: Literal["MORDRED", "RDKIT", "MORGAN_FP"] = "MORDRED"
    data: Dict[str, str]

    @validator("data", always=True)
    def validate_data(cls, dat):
        """
        Validates the given substances.
        """
        for name, smiles in dat.items():
            if not is_valid_smiles(smiles):
                raise StrictValidationError(
                    f"The SMILES '{smiles}' for molecule '{name}' does "
                    f"not appear to be valid."
                )
        return dat

    @property
    def values(self) -> list:
        """
        Returns the labels of the given set of molecules.
        """
        # Since the order of dictionary keys is important here, this will only work
        # for Python 3.7 or higher
        return list(self.data.keys())

    @cached_property
    def comp_df(self) -> pd.DataFrame:
        """
        See base class.
        """
        vals = list(self.data.values())
        pref = self.name + "_"

        # Get the raw fingerprints
        if self.encoding == "MORDRED":
            comp_df = smiles_to_mordred_features(vals, prefix=pref)
        elif self.encoding == "RDKIT":
            comp_df = smiles_to_rdkit_features(vals, prefix=pref)
        elif self.encoding == "MORGAN_FP":
            comp_df = smiles_to_fp_features(vals, prefix=pref)

        # Drop NaN and constant columns
        comp_df = comp_df.loc[:, ~comp_df.isna().any(axis=0)]
        comp_df = df_drop_single_value_columns(comp_df)

        # If there are bool columns, convert them to int (possible for Mordred)
        comp_df.loc[:, comp_df.dtypes == bool] = comp_df.loc[
            :, comp_df.dtypes == bool
        ].astype(int)

        # Label the rows with the molecule names
        comp_df.index = self.values

        # Get a decorrelated subset of the fingerprints
        if self.decorrelate:
            if isinstance(self.decorrelate, bool):
                comp_df = df_uncorrelated_features(comp_df)
            else:
                comp_df = df_uncorrelated_features(comp_df, threshold=self.decorrelate)

        return comp_df


# Available encodings for substance parameters
SUBSTANCE_ENCODINGS = get_args(get_type_hints(GenericSubstance)["encoding"])


class Custom(DiscreteParameter):
    """
    Parameter class for custom parameters where the user can read in a precomputed
    representation for labels, e.g. from quantum chemistry.
    """

    # object variables
    encoding = "CUSTOM"
    decorrelate: Union[StrictBool, confloat(gt=0.0, lt=1.0, strict=True)] = True
    data: pd.DataFrame

    @validator("data")
    def validate_data(cls, data, values):
        """
        Validates the dataframe with the custom representation.
        """
        if data.isna().any().any():
            raise StrictValidationError(
                f"The custom dataframe for parameter {values['name']} contains NaN "
                f"entries, which is not supported."
            )

        if len(data) != len(set(data.index)):
            raise StrictValidationError(
                f"The custom dataframe for parameter {values['name']} contains "
                f"duplicated indices. Please only provide dataframes with unique"
                f" indices."
            )

        # Remove zero variance and string columns
        data = df_drop_string_columns(data)
        data = df_drop_single_value_columns(data)

        return data

    @property
    def values(self) -> list:
        """
        Returns the representing labels of the parameter.
        """
        return self.data.index.to_list()

    @cached_property
    def comp_df(self) -> pd.DataFrame:
        """
        See base class.
        """
        # The encoding is directly provided by the user
        comp_df = self.data

        # Get a decorrelated subset of the provided features
        if self.decorrelate:
            if isinstance(self.decorrelate, bool):
                comp_df = df_uncorrelated_features(comp_df)
            else:
                comp_df = df_uncorrelated_features(comp_df, threshold=self.decorrelate)

        return comp_df


def parameter_cartesian_prod_to_df(
    parameters: List[Parameter],
) -> pd.DataFrame:
    """
    Creates the Cartesian product of all parameter values. Ignores continuous
    parameters.

    Parameters
    ----------
    parameters: List[Parameter]
        List of parameter objects.

    Returns
    -------
    pd.DataFrame
        A dataframe containing all possible discrete parameter value combinations.
    """
    lst_of_values = [p.values for p in parameters if p.is_discrete]
    lst_of_names = [p.name for p in parameters if p.is_discrete]
    if len(lst_of_names) < 1:
        return pd.DataFrame()

    index = pd.MultiIndex.from_product(lst_of_values, names=lst_of_names)
    ret = pd.DataFrame(index=index).reset_index()

    return ret


# TODO: self.values could be renamed into something else since it's clashing with
#  pydantic enforced syntax, for instance 'labels' (but that's weird for numeric
#  discrete parameters)

# TODO: self.values could be a variable of the base class since it's shared between all
#  parameter. It's essentially the list of labels, always one dimensional
