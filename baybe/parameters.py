# pylint: disable=R0903,W0235,E0401
"""
Functionality to deal wth different parameters
"""
from abc import ABC, abstractmethod

import numpy as np

import pandas as pd

_allowed_types = [
    "generic",
    "numeric_discrete",
    "numeric_continuous",
    "categorical",
    "substance_generic",
]


class GenericParameter(ABC):
    """
    Base class for different Parameters. Will handle storing info about the type,
    range, constraints and in-range checks
    """

    def __init__(self, name: str = "Parameter"):
        self.type = "generic"
        self.name = name

    @abstractmethod
    def is_in_range(self, item: object):
        """
        Tells whether an item is within the current parameter range. Its true by default
        since the parameter is assumed unbound, but overridden by derived parameter
        classes
        """
        return True


class NumericDiscrete(GenericParameter):
    """
    Parameter class for numerical but discrete parameters (aka setpoints)
    """

    def __init__(self, values: list = None, input_tolerance=0.0):
        super().__init__()

        self.type = "numeric_discrete"
        self.values = [] if values is None else values

        # allowed experimental uncertainty when reading in measured values
        self.input_tolerance = input_tolerance

    def is_in_range(self, item: float):
        differences_acceptable = [
            np.abs(bla - item) <= self.input_tolerance for bla in self.values
        ]
        if any(differences_acceptable):
            return True

        return False


class Categorical(GenericParameter):
    """
    Parameter class for categorical parameters
    """

    def __init__(self, labels: list = None):
        super().__init__()

        self.type = "categorical"
        self.labels = [] if labels is None else labels

    def is_in_range(self, item: str):
        if item in self.labels:
            return True

        return False


class GenericSubstance(GenericParameter):
    """
    Parameter class for generic substances that will be treated with Mordred+PCA
    """

    def __init__(self, labels: list = None, smiles: list = None):
        super().__init__()

        self.type = "substance_generic"
        self.labels = [] if labels is None else labels
        self.smiles = [] if smiles is None else smiles

    def is_in_range(self, item: str):
        if item in self.labels:
            return True

        return False


class Custom(GenericParameter):
    """
    Parameter class for custom parameters where the user can read in a precomputed
    representation for labels, e.g. from quantum chemistry
    """

    def __init__(self, labels: list = None, repesentation: pd.DataFrame = None):
        super().__init__()

        self.type = "custom"
        self.labels = [] if labels is None else labels
        self.representation = (
            pd.DataFrame([]) if repesentation is None else repesentation
        )

    def is_in_range(self, item: str):
        if item in self.labels:
            return True

        return False


class NumericContinuous(GenericParameter):
    """
    Parameter class for numerical parameters that are continuous
    """

    def __init__(self, lbound: float, ubound: float):
        super().__init__()

        self.lbound = lbound  # lower bound
        self.ubound = ubound  # upper bound

    def is_in_range(self, item: float):
        return self.lbound <= item <= self.ubound
