"""
Functionality for different experimental parameter types.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from typing import ClassVar, Dict, List, Literal, Optional, Type, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, confloat, Extra, StrictBool, validator
from sklearn.metrics.pairwise import pairwise_distances

from .utils import (
    check_if_in,
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


class Parameter(ABC, BaseModel, extra=Extra.forbid, arbitrary_types_allowed=True):
    """
    Abstract base class for all parameters. Stores information about the
    type, range, constraints, etc. and handles in-range checks, transformations etc.
    """

    # class variables
    type: ClassVar[str]
    requires_encoding: ClassVar[str]
    SUBCLASSES: ClassVar[Dict[str, Type[Parameter]]] = {}

    # object variables
    name: str

    @classmethod
    def create(cls, config: dict) -> Parameter:
        """Creates a new parameter object matching the given specifications."""
        config = config.copy()
        param_type = config.pop("type")
        check_if_in(param_type, list(Parameter.SUBCLASSES.keys()))
        return cls.SUBCLASSES[param_type](**config)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Registers new subclasses dynamically."""
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.type] = cls

    def is_in_range(self, item: object) -> bool:
        """
        Tells whether an item is within the parameter range.
        """
        # TODO: in terms of coding style, this is not ideal: `values` is currently only
        #  defined in the subclasses but not in the base class since it is either a
        #  member or a property, depending on the parameter type --> better solution?
        return item in self.values

    @property
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
        if self.requires_encoding:
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


class Categorical(Parameter):
    """
    Parameter class for categorical parameters.
    """

    # class variables
    type = "CAT"
    requires_encoding = True

    # object variables
    values: list
    encoding: Literal["OHE", "INT"] = "OHE"

    # validators
    _validated_values = validator("values", allow_reuse=True)(_validate_value_list)

    @property
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


class NumericDiscrete(Parameter):
    """
    Parameter class for discrete numerical parameters (a.k.a. setpoints).
    """

    # class variables
    type = "NUM_DISCRETE"
    requires_encoding = False

    # object variables
    values: list
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

    @property
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


class GenericSubstance(Parameter):
    """
    Parameter class for generic substances that are treated with cheminformatics
    descriptors.

    Only a decorrelated subset of descriptors should be used as otherwise this can
    result in a large number of features. For a handful of molecules, keeping only
    descriptors that have a maximum correlation of 0.7 reduces the number of
    descriptors to about 5-20. The number might be substantially higher with more
    labels given.
    """

    # class variables
    type = "SUBSTANCE"
    requires_encoding = True

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
                raise ValueError(
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

    @property
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


class Custom(Parameter):
    """
    Parameter class for custom parameters where the user can read in a precomputed
    representation for labels, e.g. from quantum chemistry.
    """

    # class variables
    type = "CUSTOM"
    requires_encoding = True

    # object variables
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

    @property
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
    Creates the Cartesion product of all parameter values.

    Parameters
    ----------
    parameters: List[Parameter]
        List of parameter objects.

    Returns
    -------
    pd.DataFrame
        A dataframe containing all parameter value combinations.
    """
    lst_of_values = [p.values for p in parameters]
    lst_of_names = [p.name for p in parameters]

    index = pd.MultiIndex.from_product(lst_of_values, names=lst_of_names)
    ret = pd.DataFrame(index=index).reset_index()

    return ret


def scaled_view(
    data_fit: Union[pd.DataFrame, pd.Series],
    data_transform: Union[pd.DataFrame, pd.Series],
    parameters: Optional[List[Parameter]] = None,
    scalers: Optional[dict] = None,
) -> pd.DataFrame:
    """
    # TODO: Revise this function and its docstring. (Currently, the function is not used
    #   at all. Hence, revision has been skipped in the code release process.)

    Comfort function to scale data given different scaling methods for different
    parameter types.

    Parameters
    ----------
    data_fit : Union[pd.DataFrame, pd.Series]
        Data on which the scalers are fit.
    data_transform : Union[pd.DataFrame, pd.Series]
        Data to be transformed.
    parameters : Optional[List[Parameter]]
        List of baybe parameter objects.
    scalers : Optional[dict]
        Dict with parameter types as keys and sklearn scaler objects as values.

    Returns
    -------
    pd.DataFrame
        The scaled parameter view.

    Examples
    --------
    scalers = {"NUM_DISCRETE": StandardScaler(), "CAT": None}
    scaled_data = scaled_view(
        data_fit = searchspace_comp_rep,
        data_transform = measurements_comp_rep,
        parameters = parameters,
        scalers = scalers,
    )
    """
    transformed = data_transform.copy()
    if parameters is None:
        log.warning("No parameters were provided, not performing any scaling")
        return transformed
    if scalers is None:
        scalers = {}

    for param in parameters:
        if param.comp_df is None:
            # Instead of enforcing this one could automatically detect columns based
            # on the starting of the name.
            raise AttributeError(
                "You are trying to scale parameters that have never used the "
                "transformation from experimental to computational representation. "
                "This means the needed columns cannot be identified."
            )

        # If no scaling instructions provided skip scaling
        if (param.type not in scalers) or (scalers.get(param.type) is None):
            continue

        scaler = scalers.get(param.type)
        if len(param.comp_df.columns) == 1:
            scaler.fit(data_fit[param.comp_df.columns].values.reshape(-1, 1))

            transformed[param.comp_df.columns] = scaler.transform(
                data_transform[param.comp_df.columns].values.reshape(-1, 1)
            )
        else:
            scaler.fit(data_fit[param.comp_df.columns].values)

            transformed[param.comp_df.columns] = scaler.transform(
                data_transform[param.comp_df.columns].values
            )

    return transformed


# TODO: self.values could be renamed into something else since it's clashing with
#  pydantic enforced syntax, for instance 'labels' (but that's weird for numeric
#  discrete parameters)

# TODO: self.values could be a variable of the base class since it's shared between all
#  parameter. It's essentially the list of labels, always one dimensional
