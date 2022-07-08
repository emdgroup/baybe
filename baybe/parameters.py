"""
Functionality to deal wth different parameters
"""
import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd

allowed_types = ["CAT", "NUM_DISCRETE"]
# allowed_types = ["NUM_DISCRETE", "NUM_CONTINUOUS", "CAT", "GEN_SUBSTANCE", "CUSTOM"]
allowed_encodings = ["OHE", "Integer"]

log = logging.getLogger(__name__)


class GenericParameter(ABC):
    """
    Abstract base class for different Parameters. Will handle storing info about the
    type, range, constraints and in-range checks
    """

    def __init__(self, name: str = "Parameter"):
        self.type = "GENERIC"
        self.name = name
        super().__init__()

    def __str__(self):
        string = f"Generic parameter\n" f"   Name: '{self.name}'"
        return string

    @abstractmethod
    def is_in_range(self, item: object):
        """
        Tells whether an item is within the current parameter range. Its true by default
        since the parameter is assumed unbound, but overridden by derived parameter
        classes
        """
        return True

    @classmethod
    @abstractmethod
    def from_dict(cls, dat: dict):
        """
        Creates a parameter of this type from a dictionary
        :param dat: parameter dictionary
        :return: class object
        """
        param_name = dat.get("Name", "Unnamed Parameter")
        return cls(name=param_name)

    @abstractmethod
    def transform_rep_exp2comp(self, series: pd.DataFrame = None):
        """
        Takes a pandas series in experimental representation and transforms it into the
        computational representation
        :param series: pandas series in experimental representation
        :return: None (since this si abstract method of base class)
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
        super().__init__(name)

        self.type = "CAT"
        self.values = [] if values is None else values
        self.encoding = encoding

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
        if item in self.values:
            return True

        return False

    @classmethod
    def from_dict(cls, dat):
        param_name = dat.get("Name", "Unnamed Parameter")
        param_values = dat.get("Values", [])
        param_encoding = dat.get("Encoding", "OHE")

        return cls(name=param_name, values=param_values, encoding=param_encoding)

    def transform_rep_exp2comp(self, series: pd.DataFrame = None):
        """bla"""

        if self.encoding == "OHE":
            data = {}
            for value in self.values:
                coldata = []
                for itm in series.values:
                    if itm not in self.values:
                        raise ValueError(
                            f"Value {itm} is not in list of permitted values "
                            f"{self.values} for parameter {self.name}"
                        )
                    if itm == value:
                        coldata.append(1)
                    else:
                        coldata.append(0)
                    data[f"{self.name}_encoded_val_{value}"] = coldata
            transformed = pd.DataFrame(data)

        elif self.encoding == "Integer":
            mapping = {val: k for k, val in enumerate(self.values)}
            coldata = []

            for itm in series.values:
                if itm not in self.values:
                    raise ValueError(
                        f' "Value {itm} is not in list of permitted values'
                        f" {self.values} for parameter {self.name}"
                    )
                coldata.append(mapping[itm])

            transformed = pd.DataFrame(coldata, columns=[f"{self.name}_encoded"])
        else:
            raise ValueError(
                f"Parameter {self.name} has encoding {self.encoding} specified, "
                f"but encoding must be one of {allowed_encodings}."
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
        super().__init__(name)

        self.type = "NUM_DISCRETE"

        self.values = [] if values is None else values
        if len(self.values) < 2:
            raise AssertionError(
                f"Numerical arameter {self.name} must have at least 2 " f"unqiue values"
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
        param_name = dat.get("Name", "Unnamed Parameter")
        param_values = dat.get("Values", [])
        param_tolerance = dat.get("Tolerance", 0.0)

        return cls(
            name=param_name, values=param_values, input_tolerance=param_tolerance
        )

    def transform_rep_exp2comp(self, series: pd.DataFrame = None):
        """
        Takes a pandas series in experimental representation and transforms it into the
        computational representation
        :param series: pandas series in experimental representation
        :return: transformed: pandas dataframe with data in comp representation
        """

        return series


class GenericSubstance(GenericParameter):
    """
    Parameter class for generic substances that will be treated with Mordred+PCA
    """

    def __init__(
        self, name: str = "Unnamed Parameter", values: list = None, smiles: list = None
    ):
        super().__init__(name=name)

        self.type = "GEN_SUBSTANCE"
        self.values = [] if values is None else values
        self.smiles = [] if smiles is None else smiles

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
        super().__init__(name=name)

        self.type = "CUSTOM"
        self.values = [] if values is None else values
        self.representation = (
            pd.DataFrame([]) if repesentation is None else repesentation
        )

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
        super().__init__(name=name)
        self.type = "NUM_CONTINUOUS"
        self.lbound = -np.inf if lbound is None else lbound  # lower bound
        self.ubound = np.inf if ubound is None else ubound  # upper bound

    def is_in_range(self, item: float):
        return self.lbound <= item <= self.ubound


def parse_parameter(param_dict: dict = None) -> GenericParameter:
    """Parses a dictionary into a parameter class object"""
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
    parameters: iteratable
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
