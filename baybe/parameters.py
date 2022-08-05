"""
Functionality to deal wth different parameters
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from typing import ClassVar, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Extra, validator
from sklearn.metrics.pairwise import pairwise_distances

from .utils import check_if_in

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
            f"This would cause duplicates in the possible experiments."
        )
    return lst


class Parameter(ABC, BaseModel, extra=Extra.forbid, arbitrary_types_allowed=True):
    """
    Abstract base class for all parameters. Stores information about the
    type, range, constraints, etc. and handles in-range checks, transformations etc.
    """

    # class variables
    type: ClassVar[str]
    SUBCLASSES: ClassVar[Dict[str, Parameter]] = {}

    # object variables
    name: str

    # TODO: this becomes obsolete in pydantic 2.0 when the __post_init_post_parse__
    #   is available:
    #   - https://github.com/samuelcolvin/pydantic/issues/691
    #   - https://github.com/samuelcolvin/pydantic/issues/1729
    comp_cols: list = []
    comp_values: Optional[np.ndarray] = None

    @classmethod
    def create(cls, config: dict) -> Parameter:
        """Creates a new parameter object matching the given specifications."""
        config = config.copy()
        param_type = config.pop("type")
        check_if_in(param_type, list(Parameter.SUBCLASSES.keys()))
        return cls.SUBCLASSES[param_type](**config)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.type] = cls

    @abstractmethod
    def is_in_range(self, item: object) -> bool:
        """
        Tells whether an item is within the parameter range.
        """

    @abstractmethod
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


class Categorical(Parameter):
    """
    Parameter class for categorical parameters.
    """

    # class variables
    type = "CAT"

    # object variables
    values: list
    encoding: Literal["OHE", "INT"] = "OHE"

    # validators
    _validated_values = validator("values", allow_reuse=True)(_validate_value_list)

    def is_in_range(self, item: str) -> bool:
        """
        See base class.
        """
        return item in self.values

    def transform_rep_exp2comp(self, data: pd.Series = None) -> pd.DataFrame:
        """
        See base class.
        """
        # IMPROVE neater implementation eg via vectorized mapping

        if self.encoding == "OHE":
            transformed_data = {}
            for value in self.values:
                coldata = []
                for itm in data.values:
                    if itm not in self.values:
                        raise ValueError(
                            f"Value {itm} is not in list of permitted values "
                            f"{self.values} for parameter {self.name}"
                        )
                    if itm == value:
                        coldata.append(1)
                    else:
                        coldata.append(0)

                    colname = f"{self.name}_encoded_val_{value}"
                    transformed_data[colname] = coldata
                    if colname not in self.comp_cols:
                        self.comp_cols.append(colname)

            transformed = pd.DataFrame(transformed_data)

        elif self.encoding == "INT":
            mapping = {val: k for k, val in enumerate(self.values)}
            coldata = []

            for itm in data.values:
                if itm not in self.values:
                    raise ValueError(
                        f' "Value {itm} is not in list of permitted values'
                        f" {self.values} for parameter {self.name}"
                    )
                coldata.append(mapping[itm])

            colname = f"{self.name}_encoded"
            transformed = pd.DataFrame(coldata, columns=[colname])
            if colname not in self.comp_cols:
                self.comp_cols.append(colname)

        return transformed


class NumericDiscrete(Parameter):
    """
    Parameter class for discrete numerical parameters (a.k.a. setpoints).
    """

    # class variables
    type = "NUM_DISCRETE"

    # object variables
    values: list
    tolerance: float

    # validators
    _validated_values = validator("values", allow_reuse=True)(_validate_value_list)

    @validator("tolerance")
    def validate_tolerance(cls, tolerance, values):
        """
        Validates that the tolerance (i.e. allowed experimental uncertainty when
        reading in measured values) is safe. A tolerance larger than half the minimum
        distance between parameter values is not allowed because that could cause
        ambiguity when inputting datapoints later.
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

    def is_in_range(self, item: float) -> bool:
        """
        See base class.
        """
        differences_acceptable = [
            np.abs(bla - item) <= self.tolerance for bla in self.values
        ]
        return any(differences_acceptable)

    def transform_rep_exp2comp(self, data: pd.Series = None) -> pd.DataFrame:
        """
        See base class.
        """

        # Comp column is identical with the experimental columns
        self.comp_cols = [self.name]
        self.comp_values = data.values

        # There is nothing to transform for this parameter type
        return data.rename(self.name).to_frame()


class GenericSubstance(Parameter, ABC):
    """
    Parameter class for generic substances that are treated with Mordred+PCA.
    """

    # class variables
    type = "GEN_SUBSTANCE"

    # object variables
    encoding: Literal["MORDRED", "RDKIT", "MORGAN_FP"]


class Custom(Parameter, ABC):
    """
    Parameter class for custom parameters where the user can read in a precomputed
    representation for labels, e.g. from quantum chemistry.
    """

    # class variables
    type = "CUSTOM"

    # object variables
    data: pd.DataFrame
    identifier_col_idx: int = 0
    values: Optional[list] = None

    @validator("data")
    def validate_data(cls, data, values):
        """
        Validates the dataframe with the custom representaton
        """
        if data.isna().any().any():
            # TODO Tried to trigger this with a Nan, but for some reason the Exception
            #  is not raised. This leads to an error in the identifier_col_idx validator
            #  since the 'data' entry is not in the dict
            raise ValueError(
                f"The custom dataframe for parameter {values['name']} contains NaN "
                f"entries, this is not supported"
            )

        return data

    @validator("identifier_col_idx", always=True)
    def validate_identifier_col(cls, col, values):
        """
        Validates the column index which identifies the label column
        """
        if (col < 0) or (col >= len(values["data"].columns)):
            raise ValueError(
                f"identifier_col_idx was {col} but must be a column index between "
                f"(inclusive) 0 and {len(values['data'].columns)-1}"
            )

        return col

    @validator("values", always=True)
    def validate_values(cls, vals, values):
        """
        Initializes the representing labels for this parameter
        """
        if vals is not None:
            raise NotImplementedError(
                "Currently with pydantic we declare 'values' as class variable but it "
                "is a non-validated variable and should not be provided in the config"
            )

        data = values["data"]
        idx_col = values["identifier_col_idx"]

        return data.iloc[:, idx_col].to_list()

    def is_in_range(self, item: object) -> bool:
        """
        See base class.
        """
        return item in self.data.iloc[:, self.identifier_col_idx].to_list()

    def transform_rep_exp2comp(self, data: pd.Series = None) -> pd.DataFrame:
        """
        See base class.
        """

        # Comp columns and columns names
        mergecol = self.data.columns[self.identifier_col_idx]
        self.comp_cols = self.data.drop(columns=mergecol).columns.to_list()
        self.comp_values = self.data.drop(columns=mergecol).values

        # Transformation is just lookup for this parameter type
        transformed = pd.merge(
            left=data.rename(mergecol).to_frame(),
            right=self.data,
            on=mergecol,
            how="left",
        ).drop(columns=mergecol)

        return transformed


class NumericContinuous(Parameter, ABC):
    """
    Parameter class for continuous numerical parameters.
    """

    # class variables
    type = "NUM_CONTINUOUS"


def parameter_outer_prod_to_df(
    parameters: List[Parameter],
) -> pd.DataFrame:
    """
    Creates the Cartesion product of all parameter values (ignoring non-discrete
    parameters).

    Parameters
    ----------
    parameters: List[Parameter]
        List of parameter objects.

    Returns
    -------
    pd.DataFrame
        A dataframe containing all parameter value combinations.
    """
    allowed_types = Parameter.SUBCLASSES
    lst_of_values = [p.values for p in parameters if p.type in allowed_types]
    lst_of_names = [p.name for p in parameters if p.type in allowed_types]

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
        if (param.comp_cols is None) or (len(param.comp_cols) < 1):
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
        if len(param.comp_cols) == 1:
            scaler.fit(data_fit[param.comp_cols].values.reshape(-1, 1))

            transformed[param.comp_cols] = scaler.transform(
                data_transform[param.comp_cols].values.reshape(-1, 1)
            )
        else:
            scaler.fit(data_fit[param.comp_cols].values)

            transformed[param.comp_cols] = scaler.transform(
                data_transform[param.comp_cols].values
            )

    return transformed
