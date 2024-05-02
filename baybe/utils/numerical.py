"""Utilities for numeric operations."""

import os
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from baybe.utils.boolean import strtobool

VARNAME_NUMPY_USE_SINGLE_PRECISION = "BAYBE_NUMPY_USE_SINGLE_PRECISION"
"""Environment variable name for enforcing single precision in numpy."""

DTypeFloatNumpy = (
    np.float32
    if strtobool(os.environ.get(VARNAME_NUMPY_USE_SINGLE_PRECISION, "False"))
    else np.float64
)
"""Floating point data type used for numpy arrays."""

DTypeFloatONNX = np.float32
"""Floating point data type used for ONNX models.

Currently, ONNX runtime does not seem to have full support for double precision.
There is no clear documentation but some references can be found here (version 1.16.0):

 * https://onnx.ai/sklearn-onnx/auto_tutorial/plot_abegin_convert_pipeline.html#converts-the-model
 * https://onnx.ai/sklearn-onnx/auto_tutorial/plot_ebegin_float_double.html
"""  # noqa: E501


def geom_mean(arr: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    """Calculate the (weighted) geometric mean along the second axis of a 2-D array.

    Alternative to ``gmean`` from scipy that avoids logarithms and division errors.

    Args:
        arr: The array containing the values for the mean computation.
        weights: Weights for the mean computation.

    Returns:
        A 1-D array containing the row-wise geometric means of the given array.
    """
    return np.prod(np.power(arr, np.atleast_2d(weights) / np.sum(weights)), axis=1)


def closest_element(array: npt.ArrayLike, target: float) -> float:
    """Find the element of an array that is closest to a target value.

    Args:
        array: The array in which the closest value is to be found.
        target: The target value.

    Returns:
        The closest element.
    """
    if np.ndim(array) == 0:
        return np.asarray(array).item()
    array = np.ravel(array)
    return array[np.abs(array - target).argmin()]


def closer_element(x: float, y: float, target: float) -> float:
    """Determine which of two given inputs is closer to a target value.

    Args:
        x: The first input that should be checked.
        y: The second input that should be checked.
        target: The target value.

    Returns:
        The closer of the two elements.
    """
    return x if np.abs(x - target) < np.abs(y - target) else y
