"""
Functions for bound transforms.
"""
import numpy as np
from numpy.typing import ArrayLike


def bound_linear(
    arr: ArrayLike, lower: float, upper: float, descending: bool
) -> np.ndarray:
    """
    A function that linearly maps input values in a specified interval
    [lower, upper] to the interval [0, 1]. Outside the specified interval, the function
    remains constant (that is, 0 or 1, depending on the side and selected mode).

    Parameters
    ----------
    arr : ArrayLike
        The values to be mapped.
    lower : float
        The lower boundary of the linear mapping interval.
    upper : float
        The upper boundary of the linear mapping interval.
    descending : bool
        If True, the function values decrease from 1 to 0 in the specified interval.
        If False, they increase from 0 to 1. Outside the interval, the boundary function
        values are extended.

    Returns
    -------
    np.ndarray
        An array containing the transformed values.
    """
    arr = np.array(arr)
    if descending:
        res = (upper - arr) / (upper - lower)
        res[arr > upper] = 0.0
        res[arr < lower] = 1.0
    else:
        res = (arr - lower) / (upper - lower)
        res[arr > upper] = 1.0
        res[arr < lower] = 0.0

    return res


def bound_triangular(arr: ArrayLike, lower: float, upper: float) -> np.ndarray:
    """
    A "triangular" function that is 0 outside a specified interval and linearly
    increases to 1 from both interval ends, reaching the value 1 at the center of the
    interval.

    Parameters
    ----------
    arr : ArrayLike
        The values to be mapped.
    lower : float
        The lower end of the triangle interval. Below, the mapped values are 0.
    upper : float
        The upper end of the triangle interval. Above, the mapped values are 0.

    Returns
    -------
    np.ndarray
        An array containing the transformed values.
    """
    mid = lower + (upper - lower) / 2
    arr = np.array(arr)
    res = (arr - lower) / (mid - lower)
    res[arr > mid] = (upper - arr[arr > mid]) / (upper - mid)
    res[arr > upper] = 0.0
    res[arr < lower] = 0.0

    return res


def bound_bell(arr: ArrayLike, lower: float, upper: float) -> np.ndarray:
    """
    A Gaussian bell curve, specified through the boundary values of the sigma interval.
    Reaches the maximum value of 1 at the interval center.

    Parameters
    ----------
    arr : ArrayLike
        The values to be mapped.
    lower : float
        The input value corresponding to the upper sigma interval boundary.
    upper : float
        The input value corresponding to the lower sigma interval boundary.

    Returns
    -------
    np.ndarray
        An array containing the transformed values.
    """
    mean = np.mean([lower, upper])
    std = (upper - lower) / 2
    res = np.exp(-((arr - mean) ** 2) / (2.0 * std**2))

    return res
