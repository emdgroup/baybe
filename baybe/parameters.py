"""
Functionality to deal wth different parameters
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Extra, validator

from baybe.utils import check_if_in

log = logging.getLogger(__name__)


class ParameterConfig(BaseModel, extra=Extra.forbid):
    """Configuration class for creating parameter objects."""

    name: str
    type: str
    values: list
    tolerance: Optional[float]  # TODO: conditional validation depending on type
    encoding: Optional[str]

    @validator("type")
    def validate_type(cls, val):
        """Validates if the given parameter type exists."""
        check_if_in(val, Parameter.SUBCLASSES)
        return val

    @validator("encoding", always=True)
    def validate_encoding(cls, val, values):
        """Validates, for parameters that require an encoding, if an encoding is
        provided and if the selection is possible for the parameter type."""
        if values["type"] in Parameter.ENCODINGS:
            if val is None:
                raise ValueError(
                    f"For parameter '{values['name']}' of type {values['type']}, an "
                    f"encoding must be specified. Select one of "
                    f"{Parameter.ENCODINGS[values['type']]}. "
                )
            check_if_in(val, Parameter.ENCODINGS[values["type"]])
        return val


class Parameter(ABC):
    """
    Abstract base class for different parameters. Will handle storing info about the
    type, range, constraints and in-range checks, transformations etc
    """

    TYPE: str
    SUBCLASSES: Dict[str, Parameter] = {}
    ENCODINGS: Dict[Parameter, List[str]] = {}

    def __init__(self, config: ParameterConfig):
        self.name = config.name
        self.values = config.values or []
        self.comp_cols = []
        self.comp_values = None

    def __str__(self):
        string = (
            f"Generic parameter\n"
            f"   Name:     '{self.name}'\n"
            f"   Values:   {self.values}\n"
        )
        return string

    @classmethod
    # TODO: add type hint once circular import problem has been fixed
    def create(cls, config) -> Parameter:
        """Creates a new parameter object matching the given specifications."""
        return cls.SUBCLASSES[config.type](config)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.TYPE] = cls
        if hasattr(cls, "ALLOWED_ENCODINGS"):
            cls.ENCODINGS[cls.TYPE] = cls.ALLOWED_ENCODINGS

    @abstractmethod
    def is_in_range(self, item: object):
        """
        Tells whether an item is within the current parameter range.
        """
        return True

    @classmethod
    def from_dict(cls, config_dict: dict) -> Parameter:
        """Creates a parameter from a config dictionary."""
        return cls(ParameterConfig(**config_dict))

    @abstractmethod
    def transform_rep_exp2comp(self, data: pd.DataFrame = None):
        """
        Function to transform data from the experimental to computational representation

        Parameters
        ----------
        data: pd.DataFrame
            Data to be transformed
        """
        return None


class Categorical(Parameter):
    """
    Parameter class for categorical parameters
    """

    TYPE = "CAT"
    ALLOWED_ENCODINGS = ["OHE", "Integer"]

    def __init__(self, config: ParameterConfig):
        super().__init__(config)

        self.encoding = config.encoding
        self.comp_cols = []

        if len(self.values) != len(np.unique(self.values)):
            raise ValueError(
                f"Values for parameter {self.name} are not unique. This would cause "
                f"duplicates in the possible experiments."
            )

    def __str__(self):
        string = (
            f"Categorical parameter\n"
            f"   Name:     '{self.name}'\n"
            f"   Values:   {self.values}\n"
            f"   Encoding: {self.encoding}"
        )

        return string

    def is_in_range(self, item: str):
        """
        See base class
        """
        if item in self.values:
            return True

        return False

    def transform_rep_exp2comp(self, data: pd.DataFrame = None):
        """
        See base class
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

        elif self.encoding == "Integer":
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
    Parameter class for numerical but discrete parameters (aka setpoints)
    """

    TYPE = "NUM_DISCRETE"

    def __init__(self, config: ParameterConfig):
        super().__init__(config)

        self.comp_cols = []
        self.comp_values = None

        if len(self.values) < 2:
            raise AssertionError(
                f"Numerical parameter {self.name} must have at least 2 "
                f"unqiue values"
            )

        # allowed experimental uncertainty when reading in measured values
        # if the requested tolerance is larger than half the minimum distance between
        # parameter values a warning is printed because that could cause ambiguity when
        # inputting datapoints later
        max_tol = (
            np.min(
                np.abs(
                    [
                        config.values[k] - config.values[k - 1]
                        for k in range(1, len(config.values))
                    ]
                )
            )
            / 2.0
        )
        if config.tolerance >= max_tol:
            log.warning(
                "Parameter %s is initialized with tolerance %s, but due to the "
                "values %s a maximum tolerance of %s is suggested to avoid ambiguity.",
                config.name,
                config.tolerance,
                config.values,
                max_tol,
            )
        self.tolerance = config.tolerance

    def is_in_range(self, item: float):
        """
        See base class
        """
        differences_acceptable = [
            np.abs(bla - item) <= self.tolerance for bla in self.values
        ]
        if any(differences_acceptable):
            return True

        return False

    def __str__(self):
        string = (
            f"Numerical discrete parameter\n"
            f"   Name:           '{self.name}'\n"
            f"   Values:          {self.values}\n"
            f"   Input Tolerance: {self.tolerance}"
        )

        return string

    def transform_rep_exp2comp(self, data: pd.DataFrame = None):
        """
        See base class
        """

        # Comp column is identical with the experimental columns
        self.comp_cols = [self.name]
        self.comp_values = data.values

        # There is nothing to transform for this parameter type
        return data


class GenericSubstance(Parameter):
    """
    Parameter class for generic substances that will be treated with Mordred+PCA
    """

    TYPE = "GEN_SUBSTANCE"
    ALLOWED_ENCODINGS = ["Mordred", "RDKit", "Morgan_FP"]  # TODO: capitalize constants

    def __init__(self, config: ParameterConfig):
        super().__init__(config)
        # self.substances = {} if substances is None else substances

        raise NotImplementedError("This parameter type is not implemented yet.")


class Custom(Parameter):
    """
    Parameter class for custom parameters where the user can read in a precomputed
    representation for labels, e.g. from quantum chemistry
    """

    TYPE = "CUSTOM"

    def __init__(self, config: ParameterConfig):
        super().__init__(config)
        # self.representation = (
        #     pd.DataFrame([]) if repesentation is None else repesentation
        # )

        raise NotImplementedError("This parameter type is not implemented yet.")


class NumericContinuous(Parameter):
    """
    Parameter class for numerical parameters that are continuous
    """

    TYPE = "NUM_CONTINUOUS"

    def __init__(self, config: ParameterConfig):
        super().__init__(config)
        # self.lbound = -np.inf if lbound is None else lbound  # lower bound
        # self.ubound = np.inf if ubound is None else ubound  # upper bound

        raise NotImplementedError("This parameter type is not implemented yet.")

    def is_in_range(self, item: float):
        # return self.lbound <= item <= self.ubound
        raise NotImplementedError()


def parameter_outer_prod_to_df(
    parameters: List[Parameter],
) -> pd.DataFrame:
    """
    Creates all possible combinations for parameters and their values (ignores
    non-discrete parameters).

    Parameters
    ----------
    parameters: list
        List of parameter objects
    Returns
    -------
    ret: pd.DataFrame
        The created data frame with the combinations
    """
    allowed_types = Parameter.SUBCLASSES
    lst_of_values = [p.values for p in parameters if p.TYPE in allowed_types]
    lst_of_names = [p.name for p in parameters if p.TYPE in allowed_types]

    index = pd.MultiIndex.from_product(lst_of_values, names=lst_of_names)
    ret = pd.DataFrame(index=index).reset_index()

    return ret


def scaled_view(
    data_fit: Union[pd.DataFrame, pd.Series],
    data_transform: Union[pd.DataFrame, pd.Series],
    parameters: Optional[List[Parameter]] = None,
    scalers: Optional[dict] = None,
):
    """
    Comfort function to scale data given different scaling methods for different
    parameter types.

    Parameters
    ----------
    data_fit:
        data on which the scalers are fit
    data_transform:
        data to be transformed
    parameters:
        list with baybe parameter instances
    scalers:
        dict with parameter types as keys and sklearn scaler instances as values

    Returns
    -------
    transformed: transformed data

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
    transformed = deepcopy(data_transform)
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
