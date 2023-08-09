"""Utilities for numeric operations."""


from typing import List

import numpy as np


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


def closest_element(array: np.ndarray, target: float) -> float:
    """Finds the element of an array that is closest to a target value."""
    return array[np.abs(array - target).argmin()]


def closer_element(x: float, y: float, target: float) -> float:
    """Determines which of two given inputs is closer to a target value."""
    return x if np.abs(x - target) < np.abs(y - target) else y
