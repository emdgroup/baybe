"""
Function for initial point sampling
"""
# pylint: disable=invalid-name
# flake8: noqa

import random

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def _fps(points: pd.DataFrame, batch_quantity: int = 1, start_strategy: str = "random"):
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

    if start_strategy == "random":
        select_point_idx = random.randrange(0, len(points))

    # TODO: add more strategies

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
