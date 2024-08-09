"""A collection of point sampling algorithms."""

import warnings
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def farthest_point_sampling(
    points: np.ndarray,
    n_samples: int = 1,
    initialization: Literal["farthest", "random"] = "farthest",
) -> list[int]:
    """Select a subset of points using farthest point sampling.

    Creates a subset of a given collection of points by successively adding points with
    the largest Euclidean distance to intermediate point selections encountered during
    the algorithmic process. The mechanism used for the initial point selection is
    configurable.

    Args:
        points: The points that are available for selection, represented as a 2-D array
            of shape ``(n, k)``, where ``n`` is the number of points and ``k`` is the
            dimensionality of the points.
        n_samples: The total number of points to be selected.
        initialization: Determines how the first points are selected:
            * ``"farthest"``: The first two selected points are those with the
              largest distance. If only a single point is requested, a deterministic
              choice is made based on the point coordinates.
            * ``"random"``: The first point is selected uniformly at random.

    Returns:
        A list containing the positional indices of the selected points.

    Raises:
        ValueError: If the provided array is not two-dimensional.
        ValueError: If the array contains no points.
        ValueError: If the input space has no dimensions.
        ValueError: If an unknown method for initialization is specified.
    """
    if (n_dims := np.ndim(points)) != 2:
        raise ValueError(
            f"The provided array must be two-dimensional but the given input had "
            f"{n_dims} dimensions."
        )
    if (n_points := len(points)) == 0:
        raise ValueError("The provided array must contain at least one row.")
    if points.shape[-1] == 0:
        raise ValueError("The provided input space must be at least one-dimensional.")
    if n_samples > n_points:
        raise ValueError(
            f"The number of requested samples ({n_samples}) cannot be larger than the "
            f"total number of points provided ({n_points})."
        )

    # Catch the pathological case upfront
    if len(np.unique(points, axis=0)) == 1:
        warnings.warn("All points are identical.", UserWarning)
        return list(range(n_samples))

    # Sort the points to produce the same result regardless of the input order
    sort_idx = np.lexsort(tuple(points.T))
    points = points[sort_idx]

    # Pre-compute the pairwise distances between all points
    dist_matrix = pairwise_distances(points)

    # Avoid wrong behavior situations where all (remaining) points are duplicates
    np.fill_diagonal(dist_matrix, -np.inf)

    # Initialize the point selection
    if initialization == "random":
        selected_point_indices = [np.random.randint(0, n_points)]
    elif initialization == "farthest":
        idx_1d = np.argmax(dist_matrix)
        selected_point_indices = list(
            map(int, np.unravel_index(idx_1d, dist_matrix.shape))
        )
        if n_samples == 1:
            return [sort_idx[selected_point_indices[0]]]
    else:
        raise ValueError(f"unknown initialization recommender: '{initialization}'")

    # Initialize the list of remaining points
    remaining_point_indices = list(range(n_points))
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

    # Undo the initial point reordering
    return sort_idx[selected_point_indices].tolist()


class DiscreteSamplingMethod(Enum):
    """Available discrete sampling methods."""

    Random = "Random"
    """Random Sampling."""

    FPS = "FPS"
    """Farthest point sampling."""


def sample_numerical_df(
    df: pd.DataFrame,
    n_points: int,
    *,
    method: DiscreteSamplingMethod = DiscreteSamplingMethod.Random,
) -> pd.DataFrame:
    """Sample data points from a numerical dataframe.

    If the requested amount of points is larger than the number of available points,
    the entire dataframe will be returned as many times at it fits into the requested
    number and the specified sampling method will only return the remainder of points.

    Args:
        df: Dataframe with purely numerical entries.
        n_points: Number of points to sample.
        method: Sampling method.

    Returns:
        The sampled points.

    Raises:
        TypeError: If the provided dataframe has non-numerical content.
        ValueError: When an invalid sampling method was provided.
    """
    if any(df[col].dtype.kind not in "iufb" for col in df.columns):
        raise TypeError(
            f"'{sample_numerical_df.__name__}' only supports purely numerical "
            f"dataframes."
        )

    # Split points in trivial and sampled parts
    n_trivial, n_sampled = divmod(n_points, len(df))

    ilocs = list(range(len(df))) * n_trivial
    if n_sampled > 0:
        if method is DiscreteSamplingMethod.FPS:
            ilocs += farthest_point_sampling(df.values, n_sampled)
        elif method is DiscreteSamplingMethod.Random:
            ilocs += df.reset_index(drop=True).sample(n_sampled).index.tolist()
        else:
            raise ValueError(f"Unrecognized sampling method: '{method}'.")

    return df.iloc[ilocs]
