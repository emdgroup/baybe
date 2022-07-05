# pylint: disable=R0903,W0235,E0401,R0912
"""
Functionality to deal wth different parameters
"""
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
    def transform_rep_exp2comp(self, series: pd.DataFrame = None, do_fit: bool = False):
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
        self.scaler_is_fitted = False
        self.scaler = StandardScaler()

        if len(self.values) != len(np.unique(self.values)):
            log.warning(
                "Values for parameter %s are not unique. This will cause duplicates in "
                "the possible experiments.",
                self.name,
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

    def transform_rep_exp2comp(self, series: pd.DataFrame = None, do_fit: bool = False):
        """
        Takes a pandas series in experimental representation and transforms it into the
        computational representation
        :param series: pandas series in experimental representation
        :return: transformed: pandas dataframe
        """
        data = {}
        if self.encoding == "OHE":
            for value in self.values:
                row = []
                for itm in series.values:
                    if itm not in self.values:
                        raise ValueError(
                            f"Value {itm} is not in list of permitted values "
                            f"{self.values} for parameter {self.name}"
                        )
                    if itm == value:
                        row.append(1)
                    else:
                        row.append(0)
                    data[f"{self.name}_val_{value}"] = row
            transformed = pd.DataFrame(data)

        elif self.encoding == "Integer":
            # Map Values
            mapping = {val: k for k, val in enumerate(self.values)}
            row = []
            for itm in series.values:
                if itm not in self.values:
                    raise ValueError(
                        f' "Value {itm} is not in list of permitted values'
                        f" {self.values} for parameter {self.name}"
                    )
                row.append(mapping[itm])
            data[f"{self.name}_encoded"] = row

            # Scale values
            if do_fit:
                if self.scaler_is_fitted:
                    log.warning(
                        "Scaler for parameter %s is already fitted, refitting might "
                        "result in unwanted behavior",
                        self.name,
                    )
                self.scaler.fit(np.array(row).reshape(-1, 1))
                self.scaler_is_fitted = True

            if not self.scaler_is_fitted:
                raise AssertionError(
                    f"Scaler for parameter {self.name} is not fitted yet but needs to "
                    f"be before transforming. Check if do_fit=True on first use"
                )

            transformed = pd.DataFrame(
                self.scaler.transform(np.array(row).reshape(-1, 1)),
                columns=[series.name],
            )
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
        self, name: str = "Unnamed Parameter", values: list = None, input_tolerance=0.0
    ):
        super().__init__(name)

        self.type = "NUM_DISCRETE"
        self.values = [] if values is None else values

        # allowed experimental uncertainty when reading in measured values
        self.input_tolerance = input_tolerance
        self.scaler_is_fitted = False
        self.scaler = StandardScaler()

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

    def transform_rep_exp2comp(self, series: pd.DataFrame = None, do_fit: bool = False):
        """
        Takes a pandas series in experimental representation and transforms it into the
        computational representation
        :param series: pandas series in experimental representation
               do_fit: boolean that is true if the scaler needs to be fitted first
        :return: transformed: pandas dataframe with data in comp representation
        """
        if do_fit:
            if self.scaler_is_fitted:
                log.warning(
                    "Scaler for parameter %s is already fitted, refitting might result "
                    "in unwanted behavior",
                    self.name,
                )
            self.scaler.fit(series.values.reshape(-1, 1))
            self.scaler_is_fitted = True

        if not self.scaler_is_fitted:
            raise AssertionError(
                f"Scaler for parameter {self.name} is not fitted yet but needs to be "
                f"before transforming. Check if do_fit=True on first use"
            )

        transformed = pd.DataFrame(
            self.scaler.transform(series.values.reshape(-1, 1)), columns=[series.name]
        )
        return transformed


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


def parameter_outer_prod_to_df(parameters: list = None):
    """
    Creates all possible combinations fo parameters (ignores non-discrete parameters).
    :param parameters: List of Parameter objects
    :return: pandas dataframe corresponding to the outer product of discrete parameter
    values
    """
    lst_of_values = [p.values for p in parameters if p.type in allowed_types]
    lst_of_names = [p.name for p in parameters if p.type in allowed_types]

    index = pd.MultiIndex.from_product(lst_of_values, names=lst_of_names)

    return pd.DataFrame(index=index).reset_index()
