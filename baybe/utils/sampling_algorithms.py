"""
A collection of point sampling algorithms.
"""

from typing import Literal

import numpy as np
from sklearn.metrics import pairwise_distances


def farthest_point_sampling(
    points: np.ndarray,
    n_samples: int = 1,
    initialization: Literal["farthest", "random"] = "farthest",
) -> np.ndarray:
    """
    Farthest point sampling.

    Creates a subset of a collection of points by successively adding points with the
    largest Euclidean distance to intermediate point selections encountered during
    the algorithmic process.

    Parameters
    ----------
    points : np.ndarray
        The points that are available for selection, represented as a 2-D array whose
        first dimension corresponds to the point index.
    n_samples : int (default = 1)
        The total number of points to be selected.
    initialization : "farthest" | "random" (default = "farthest")
        Determines how the first points are selected.
        * "farthest: The first two selected points are those with the largest distance.
            If only a single point is requested, it is selected randomly from these two.
        * "random": The first point is selected uniformly at random.

    Returns
    -------
    np.ndarray
        An array containing the indices of the selected points.
    """
    # Compute the pairwise distances between all points
    dist_matrix = pairwise_distances(points)

    # Initialize the point selection subset
    if initialization == "random":
        selected_point_indices = [np.random.randint(0, len(points))]
    elif initialization == "farthest":
        idx_1d = np.argmax(dist_matrix)
        selected_point_indices = list(np.unravel_index(idx_1d, dist_matrix.shape))
        if n_samples == 1:
            return np.random.choice(selected_point_indices, 1)
    else:
        raise ValueError(f"unknown initialization strategy: '{initialization}'")

    # Initialize the list of remaining points
    remaining_point_indices = list(range(len(points)))
    for idx in selected_point_indices:
        remaining_point_indices.remove(idx)

    # Successively add the points with the largest distance
    while len(selected_point_indices) < n_samples:

        # Collect distances between selected and remaining points
        dist = dist_matrix[np.ix_(remaining_point_indices, selected_point_indices)]

        # Find for each candidate point the smallest distance to the selected points
        min_dists = np.min(dist, axis=1)

        # Choose the point with the "largest smallest distance"
        selected_point_index = remaining_point_indices[np.argmax(min_dists)]

        # Add the chosen point to the selection
        selected_point_indices.append(selected_point_index)
        remaining_point_indices.remove(selected_point_index)

    return selected_point_indices
