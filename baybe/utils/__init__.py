# pylint: disable=missing-function-docstring

"""
Collection of small utilities.
"""

import random
from dataclasses import dataclass
from typing import List, TypeVar

import numpy as np
import torch

from baybe.utils.boolean import isabstract

T = TypeVar("T")


@dataclass(frozen=True, repr=False)
class Dummy:
    """
    Placeholder element for array-like data types. Useful e.g. for detecting
    duplicates in constraints.
    """

    def __repr__(self):
        return "<dummy>"


def geom_mean(arr: np.ndarray, weights: List[float] = None) -> np.ndarray:
    """
    Calculates the (weighted) geometric mean along the second axis of a given 2-D array.
    Alternative to `gmean` from scipy that avoids logarithms and division errors.

    Parameters
    ----------
    arr : np.ndarray
        The array containing the values for the mean computation.
    weights : List[float] (optional)
        Optional weights for the mean computation.

    Returns
    -------
    np.ndarray
        A 1-D array containing the row-wise geometric means of the given array.
    """
    return np.prod(np.power(arr, np.atleast_2d(weights) / np.sum(weights)), axis=1)


def get_subclasses(cls: T, recursive: bool = True, abstract: bool = False) -> List[T]:
    """
    Returns a list of subclasses for the given class.

    Parameters
    ----------
    cls
        The base class to retrieve subclasses for.
    recursive : bool
        If True, indirect subclasses (i.e. subclasses of subclasses) are included.
    abstract : bool
        If True, abstract subclasses are included.

    Returns
    -------
    list
        A list of subclasses for the given class.
    """
    subclasses = []
    for subclass in cls.__subclasses__():

        # Append direct subclass only if it is not abstract
        if abstract or not isabstract(subclass):
            subclasses.append(subclass)

        # If requested, add indirect subclasses
        if recursive:
            subclasses.extend(get_subclasses(subclass, abstract=abstract))

    return subclasses


def closest_element(array: np.ndarray, target: float) -> float:
    """Finds the element of an array that is closest to a target value."""
    return array[np.abs(array - target).argmin()]


def closer_element(x: float, y: float, target: float) -> float:
    """Determines which of two given inputs is closer to a target value."""
    return x if np.abs(x - target) < np.abs(y - target) else y


def set_random_seed(seed: int) -> None:
    """Sets the global random seed."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
