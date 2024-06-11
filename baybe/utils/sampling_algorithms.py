"""A collection of point sampling algorithms."""
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
    """Sample points according to a farthest point heuristic.

    Creates a subset of a collection of points by successively adding points with the
    largest Euclidean distance to intermediate point selections encountered during
    the algorithmic process.

    Args:
        points: The points that are available for selection, represented as a 2D array
            whose first dimension corresponds to the point index.
        n_samples: The total number of points to be selected.
        initialization: Determines how the first points are selected. When
            ``"farthest"`` is chosen, the first two selected points are those with the
            largest distance. If only a single point is requested, it is selected
            randomly from these two. When ``"random"`` is chosen, the first point is
            selected uniformly at random.

    Returns:
        A list containing the positional indices of the selected points.

    Raises:
        ValueError: If an unknown initialization recommender is used.
    """
    # Compute the pairwise distances between all points
    dist_matrix = pairwise_distances(points)

    # Initialize the point selection subset
    if initialization == "random":
        selected_point_indices = [np.random.randint(0, len(points))]
    elif initialization == "farthest":
        idx_1d = np.argmax(dist_matrix)
        selected_point_indices = list(
            map(int, np.unravel_index(idx_1d, dist_matrix.shape))
        )
        if n_samples == 1:
            return np.random.choice(selected_point_indices, 1).tolist()
        elif n_samples < 1:
            raise ValueError(
                f"Farthest point sampling must be done with >= 1 samples, but "
                f"n_samples={n_samples} was given."
            )
    else:
        raise ValueError(f"unknown initialization recommender: '{initialization}'")

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
    replacement: bool = False,
) -> pd.DataFrame:
    """Sample data points from a data frame.

    If the requested number points is larger than the number of available points,
    sampling is always done with replacement.

    Args:
        df: Data frame with purely numerical entries.
        n_points: Number of points to sample.
        method: Sampling method.
        replacement: If sampling is done with replacement or not (has no effect on FPS).

    Returns:
        The sampled points.

    Raises:
        TypeError: If df has non-numerical content.
        ValueError: When an invalid sampling method was provided.
    """
    if any(df[col].dtype.kind not in "iufb" for col in df.columns):
        raise TypeError(
            "'sample_numerical_df' only supports purely numerical data frames."
        )

    if method is DiscreteSamplingMethod.FPS:
        if n_points >= len(df):
            ilocs = list(range(len(df)))
            ilocs += np.random.choice(ilocs, n_points - len(df)).tolist()
        else:
            ilocs = farthest_point_sampling(df.values, n_points)
        sampled = df.iloc[ilocs]
    elif method is DiscreteSamplingMethod.Random:
        sampled = df.sample(n_points, replace=replacement or (n_points > len(df)))
    else:
        raise ValueError(f"Unrecognized sampling method: '{method}'.")

    return sampled
