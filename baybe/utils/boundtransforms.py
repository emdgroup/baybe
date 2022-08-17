"""
Function for bound transforms. All functions map into the interval (0,1)
"""
# pylint: disable=invalid-name
# flake8: noqa

import numpy as np


def bound_lu_linear(x, l, u, bMin):
    """
    Performs transformation into an interval which is rising linearly from 0 to 1.
    Bounds are lower (l) and upper (u)
    """
    x = np.array(x)
    if bMin:
        res = (u - x) / (u - l)
        res[x > u] = 0.0
        res[x < l] = 1.0
    else:
        res = (x - l) / (u - l)
        res[x > u] = 1.0
        res[x < l] = 0.0

    return res


def bound_lmu_linear(x, l, u):
    """
    Performs transformation into a centered interval which starts at u, linearly rises
    to 1 at m and then linearly decreases to 0 at u.
    """
    m = l + (u - l) / 2
    x = np.array(x)
    res = (x - l) / (m - l)
    res[x > m] = (u - x[x > m]) / (u - m)
    res[x > u] = 0.0
    res[x < l] = 0.0

    return res


# Bell curve
def bound_bell(x, l, u):
    """
    Performs transformation into a centered interval which is 1 at mean and follows
    the Gaussian distribution with standard deviation std.
    """
    mean = np.mean([l, u])
    std = (u - l) / 2
    return np.exp(-((x - mean) ** 2) / (2.0 * std**2))
