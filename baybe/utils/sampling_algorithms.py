"""
Function for initial point sampling
"""
# pylint: disable=invalid-name
# flake8: noqa

import math
import random

import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import (
    DotProduct,
    ExpSineSquared,
    Matern,
    RationalQuadratic,
    RBF,
    WhiteKernel,
)
from sklearn.metrics import pairwise_distances


def get_kernel(kernel="RBF"):
    """Return a sklearn kernel object"""

    if kernel == "RBF":
        return 1.0 * RBF(1.0)

    if kernel == "white":
        return DotProduct() + WhiteKernel(noise_level=0.5)

    if kernel == "Matern":
        return 1.0 * Matern(length_scale=1.0, nu=1.5)

    if kernel == "RationalQuadratic":
        return RationalQuadratic(length_scale=1.0, alpha=1.5)

    if kernel == "ExpSineSquared":
        return ExpSineSquared(length_scale=1, periodicity=1)

    return None


def _dpp(points, batch_quantity=1, kernel="RBF", epsilon=1e-10, start_index=None):
    """
    Fast implementation of the greedy algorithm from paper
    "Fast Greedy MAP Inference for Determinantal Point Process to Improve
    Recommendation Diversity"
    reference: https://github.com/laming-chen/fast-map-dpp/blob/master/dpp_test.py

    Parameters
    ----------
    points: np.ndarray
        The features of all candidate experiments that could be conducted first.
    batch_quantity: int (default = 1)
        The number of experiments to be conducted in parallel.
    kernel: str (default = 'RBF')
        The kernel functions to tell GP model how similar two data points are.
        Currently support `RBF`, `white`, `Matern`, `RationalQuadratic` and
        `ExpSineSquared`.
    epsilon: float (default = 1E-10)
        Small positive scalar

    Returns
    -------
    The DataFrame indices of the specific experiments selected by the strategy.
    """
    kernel_matrix = get_kernel(kernel).__call__(points)
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((batch_quantity, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = []
    # selected_item = np.argmax(di2s)
    selected_item = np.random.choice(item_size) if start_index is None else start_index
    selected_items.append(selected_item)
    while len(selected_items) < batch_quantity:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return pd.Index(selected_items)


def _fps(points, batch_quantity=1, start_index=None):
    """
    Implementation of farthest point sampling.

    Parameters
    ----------
    points: np.ndarray
        The features of all candidate experiments that could be conducted first.
    batch_quantity: int (default = 1)
        The number of experiments to be conducted in parallel.
    start_strategy: str (default = 'random')
        Specify the starting point in the selection process
        Currently support `random`.

    Returns
    -------
    The DataFrame indices of the specific experiments selected by the strategy.
    """
    selected_point_indices = []
    remaining_point_indices = list(range(len(points)))

    # if start_strategy == 'random':
    #     select_point_idx = random.randrange(0, len(points))
    # elif start_strategy == 'outlier':
    #     # TODO: add more strategies
    #     select_point_idx = 0

    select_point_idx = (
        random.randrange(0, len(points)) if start_index is None else start_index
    )
    selected_point_indices.append(select_point_idx)
    remaining_point_indices.remove(select_point_idx)

    dist_matrix = pairwise_distances(points)
    while len(selected_point_indices) < batch_quantity:
        # collect distance of selected points between remaining points
        dist = dist_matrix[selected_point_indices, :][:, remaining_point_indices]
        # find for each candidate point the smallest distance to the selected points
        dists = np.min(dist, axis=0)
        # choose the next candidate with largest smallest distance
        selected_point_index = remaining_point_indices[np.argmax(dists)]

        selected_point_indices.append(selected_point_index)
        remaining_point_indices.remove(selected_point_index)

    return pd.Index(selected_point_indices)
