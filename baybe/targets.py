"""
Functionality for different type of targets
"""

import logging

import numpy as np
import pandas as pd

allowed_types = ["NUM"]
allowed_modes = ["SINGLE"]

log = logging.getLogger(__name__)


class NumericalTarget:
    """
    Class for numerical targets
    """

    def __init__(
        self, name: str = "Unnamed Target", mode: str = "Max", bounds: tuple = None
    ):
        self.name = name
        self.mode = mode

        self.type = "NUM"

        # ToDo Make sure bounds is a 2-sized vector of finite values
        self.bounds = bounds

    def __str__(self):
        string = (
            f"Numerical target\n"
            f"   Name:   '{self.name}'\n"
            f"   Mode:   '{self.mode}'\n"
            f"   Bounds: {self.bounds}"
        )
        return string

    @classmethod
    def from_dict(cls, dat: dict):
        """
        Creates a target of this type from a dictionary
        :param dat: parameter dictionary
        :return: class object
        """
        targ_name = dat.get("Name", "Unnamed Target")
        targ_mode = dat.get("Mode", "Max")
        targ_bounds = dat.get("Bounds", None)

        return cls(name=targ_name, mode=targ_mode, bounds=targ_bounds)

    def transform(self, data: pd.DataFrame):
        """
        Transform data tot he computational representation. The transformation depends
        on the target mode, e.g. minimization, maximization, matching, multi-target etc

        Parameters
        ----------
        data: pd.DataFrame
            The data to be transformed.
        Returns
        -------
        The transformed data frame
        """

        # ToDo implement transforms for bounds
        if self.mode == "Max":
            if self.bounds is not None and np.isfinite(self.bounds).all():
                # ToDo implement transform wth bounds here
                return data * 3
            return data
        if self.mode == "Min":
            if self.bounds is not None and np.isfinite(self.bounds).all():
                # ToDo implement transform wth bounds here
                return -data * 3
            return -data
        if self.mode == "Match":
            if self.bounds is not None and np.isfinite(self.bounds).all():
                # ToDo implement match transform here
                raise TypeError(
                    f"Match mode is not supported for this target named {self.name} of "
                    f"type {self.type} since it has non-finite bounds or bounds are not"
                    f" defined. Bounds need to be a finite 2-touple."
                )

            raise NotImplementedError("Match mode for targets is not implemented yet.")

        raise ValueError(
            f"The mode '{self.mode}' set for target {self.name} is not recognized. "
            f"Must be either Min, Max or Match"
        )


def parse_single_target(target_dict: dict = None) -> NumericalTarget:
    """
    Parses a dictionary into a target object in single target mode
    :param target_dict: dictionary containing the target info
    :return: target class object
    """
    if target_dict is None:
        target_dict = {}

    target_type = target_dict.get("Type", "NUM")
    if target_type == "NUM":
        target = NumericalTarget.from_dict(target_dict)
    else:
        raise ValueError(
            f"Target type {target_type} is not one of the allowed "
            f"choices: {allowed_types}",
        )

    return target
