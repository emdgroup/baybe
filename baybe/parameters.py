"""
Functionality to deal wth different parameters
"""
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
import pandas as pd

allowed_types = ["CAT", "NUM_DISCRETE"]
# allowed_types = ["NUM_DISCRETE", "NUM_CONTINUOUS", "CAT", "GEN_SUBSTANCE", "CUSTOM"]


log = logging.getLogger(__name__)


class GenericParameter(ABC):
    """
    Abstract base class for different Parameters. Will handle storing info about the
    type, range, constraints and in-range checks, transformations etc
    """

    def __init__(self, name: str = "Parameter", values: Optional[list] = None):
        self.type = "GENERIC"
        self.name = name
        self.values = [] if values is None else values
        self.comp_cols = []
        self.comp_values = None
        super().__init__()

    def __str__(self):
        string = (
            f"Generic parameter\n"
            f"   Name:     '{self.name}'\n"
            f"   Values:   {self.values}\n"
        )
        return string

    @abstractmethod
    def is_in_range(self, item: object):
        """
        Tells whether an item is within the current parameter range.
        """
        return True

    @classmethod
    @abstractmethod
    def from_dict(cls, dat: dict):
        """
        Creates a parameter of this type from a dictionary

        Parameters
        ----------
        dat: dict
            Dictionary with info describing the parameter

        Returns
        -------
            Class instance
        """
        param_name = dat.get("Name", "Unnamed Parameter")
        param_values = dat.get("Values", [])
        return cls(name=param_name, values=param_values)

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


class Categorical(GenericParameter):
    """
    Parameter class for categorical parameters
    """

    def __init__(
        self,
        name: str = "Unnamed Parameter",
        values: list = None,
        encoding: str = "OHE",
    ):
        super().__init__(name=name, values=values)

        self.type = "CAT"
        self.allowed_encodings = ["OHE", "Integer"]

        self.encoding = encoding
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

    @classmethod
    def from_dict(cls, dat):
        """
        See base class
        """
        param_name = dat.get("Name", "Unnamed Parameter")
        param_values = dat.get("Values", [])
        param_encoding = dat.get("Encoding", "OHE")

        return cls(name=param_name, values=param_values, encoding=param_encoding)

    def transform_rep_exp2comp(self, data: pd.DataFrame = None):
        """
        See base class
        """

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
        else:
            raise ValueError(
                f"Parameter {self.name} has encoding {self.encoding} specified, "
                f"but encoding must be one of {self.allowed_encodings}."
            )

        return transformed


class NumericDiscrete(GenericParameter):
    """
    Parameter class for numerical but discrete parameters (aka setpoints)
    """

    def __init__(
        self,
        name: str = "Unnamed Parameter",
        values: list = None,
        input_tolerance: float = 0.0,
    ):
        super().__init__(name=name, values=values)

        self.type = "NUM_DISCRETE"
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
            np.min([values[k] - values[k - 1] for k in range(1, len(values))]) / 2.0
        )
        if input_tolerance >= max_tol:
            log.warning(
                "Parameter %s is initialized with tolerance %s, but due to the "
                "values %s a maximum tolerance of %s is suggested to avoid ambiguity.",
                self.name,
                input_tolerance,
                self.values,
                max_tol,
            )
        self.input_tolerance = input_tolerance

    def is_in_range(self, item: float):
        """
        See base class
        """
        differences_acceptable = [
            np.abs(bla - item) <= self.input_tolerance for bla in self.values
        ]
        if any(differences_acceptable):
            return True

        return False

    def __str__(self):
        string = (
            f"Numerical discrete parameter\n"
            f"   Name:           '{self.name}'\n"
            f"   Values:          {self.values}\n"
            f"   Input Tolerance: {self.input_tolerance}"
        )

        return string

    @classmethod
    def from_dict(cls, dat):
        """
        See base class
        """
        param_name = dat.get("Name", "Unnamed Parameter")
        param_values = dat.get("Values", [])
        param_tolerance = dat.get("Tolerance", 0.0)

        return cls(
            name=param_name, values=param_values, input_tolerance=param_tolerance
        )

    def transform_rep_exp2comp(self, data: pd.DataFrame = None):
        """
        See base class
        """

        # Comp column is identical with the experimental columns
        self.comp_cols = [self.name]
        self.comp_values = data.values

        # There is nothing to transform for this parameter type
        return data


class GenericSubstance(GenericParameter):
    """
    Parameter class for generic substances that will be treated with Mordred+PCA
    """

    def __init__(
        self,
        name: str = "Unnamed Parameter",
        substances: dict = None,
    ):
        super().__init__(name=name, values=list(substances.keys()))

        self.type = "GEN_SUBSTANCE"
        self.allowed_encodings = ["Mordred", "RDKit", "Morgan_FP"]

        self.substances = {} if substances is None else substances

        raise NotImplementedError("This parameter type is not implemented yet.")

    def is_in_range(self, item: str):
        if item in self.values:
            return True

        return False


class Custom(GenericParameter):
    """
    Parameter class for custom parameters where the user can read in a precomputed
    representation for labels, e.g. from quantum chemistry
    """

    def __init__(
        self,
        name: str = "Unnamed Parameter",
        values: list = None,
        repesentation: pd.DataFrame = None,
    ):
        super().__init__(name=name, values=values)

        self.type = "CUSTOM"
        self.representation = (
            pd.DataFrame([]) if repesentation is None else repesentation
        )

        raise NotImplementedError("This parameter type is not implemented yet.")

    def is_in_range(self, item: str):
        if item in self.values:
            return True

        return False


class NumericContinuous(GenericParameter):
    """
    Parameter class for numerical parameters that are continuous
    """

    def __init__(
        self,
        name: str = "Unnamed Parameter",
        lbound: float = None,
        ubound: float = None,
    ):
        super().__init__(name=name, values=[])
        self.type = "NUM_CONTINUOUS"
        self.lbound = -np.inf if lbound is None else lbound  # lower bound
        self.ubound = np.inf if ubound is None else ubound  # upper bound

        raise NotImplementedError("This parameter type is not implemented yet.")

    def is_in_range(self, item: float):
        return self.lbound <= item <= self.ubound


def parse_parameter(param_dict: dict = None) -> GenericParameter:
    """
    Parses a dictionary into a parameter class object

    Parameters
    ----------
    param_dict: dict

    Returns
    -------
        Instance of a parameter class
    """
    if param_dict is None:
        param_dict = {}

    param_type = param_dict.get("Type", None)
    if param_type == "CAT":
        param = Categorical.from_dict(param_dict)
    elif param_type == "NUM_DISCRETE":
        param = NumericDiscrete.from_dict(param_dict)
    else:
        raise ValueError(
            f"Parameter type {param_type} is not in one of the allowed "
            f"choices: {allowed_types}",
        )

    return param


def parameter_outer_prod_to_df(
    parameters: List[GenericParameter],
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
    lst_of_values = [p.values for p in parameters if p.type in allowed_types]
    lst_of_names = [p.name for p in parameters if p.type in allowed_types]

    index = pd.MultiIndex.from_product(lst_of_values, names=lst_of_names)
    ret = pd.DataFrame(index=index).reset_index()

    return ret


def scaled_view(
    data_fit: Union[pd.DataFrame, pd.Series],
    data_transform: Union[pd.DataFrame, pd.Series],
    parameters: Optional[List[GenericParameter]] = None,
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
