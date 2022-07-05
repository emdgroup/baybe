# pylint: disable=R0903,W0235
"""
Functionality for different type of targets
"""

import logging
from abc import ABC, abstractmethod

allowed_types = ["NUM", "CAT"]
allowed_modes = ["SINGLE"]

log = logging.getLogger(__name__)


class GenericTarget(ABC):
    """
    Base class for different Targets. Will handle storing info about the type, the
    range and constraints
    """

    def __init__(self, name: str = "Target"):
        self.name = name

    def __str__(self):
        string = f"Generic target\n" f"   Name: '{self.name}'"
        return string

    @classmethod
    @abstractmethod
    def from_dict(cls, dat):
        """
        Creates a target of this type from a dictionary
        :param dat: parameter dictionary
        :return: class object
        """
        targ_name = dat.get("Name", "Unnamed Target")
        return cls(name=targ_name)


class Numerical(GenericTarget):
    """
    Class for numerical targets
    """

    def __init__(self, name: str = "Unnamed Target", bounds: tuple = None):
        super().__init__(name)

        self.type = "NUM"
        self.bounds = None if bounds is None else bounds

    def __str__(self):
        string = (
            f"Numerical target\n"
            f"   Name:   '{self.name}'\n"
            f"   Bounds: {self.bounds}"
        )
        return string

    @classmethod
    def from_dict(cls, dat):
        """
        Creates a target of this type from a dictionary
        :param dat: parameter dictionary
        :return: class object
        """
        targ_name = dat.get("Name", "Unnamed Target")
        targ_bounds = dat.get("Bounds", None)
        return cls(name=targ_name, bounds=targ_bounds)


class Categorical(GenericTarget):
    """
    Class for categorical targets
    """

    def __init__(self, name: str = "Unnamed Target", labels: list = None):
        super().__init__(name)

        self.type = "CAT"
        self.labels = [] if labels is None else labels

    def __str__(self):
        string = (
            f"Categorical target\n"
            f"   Name:   '{self.name}'\n"
            f"   Labels: {self.labels}"
        )
        return string

    @classmethod
    def from_dict(cls, dat):
        """
        Creates a target of this type from a dictionary
        :param dat: parameter dictionary
        :return: class object
        """
        targ_name = dat.get("Name", "Unnamed Target")
        targ_labels = dat.get("Labels", [])
        return cls(name=targ_name, labels=targ_labels)


def parse_single_target(target_dict: dict = None) -> GenericTarget:
    """
    Parses a dictionary into a target object in single target mode
    :param target_dict: dictionary containing the target info
    :return: target class object
    """
    if target_dict is None:
        target_dict = {}

    target_type = target_dict.get("Type", None)
    if target_type == "NUM":
        target = Numerical.from_dict(target_dict)
    elif target_type == "CAT":
        target = Categorical.from_dict(target_dict)
    else:
        raise ValueError(
            f"Target type {target_type} is not one of the allowed "
            f"choices: {allowed_types}",
        )

    return target
