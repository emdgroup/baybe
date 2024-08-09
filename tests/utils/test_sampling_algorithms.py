"""Tests for sampling algorithm utilities."""

import math

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given
from sklearn.metrics import pairwise_distances

from baybe.utils.sampling_algorithms import (
    DiscreteSamplingMethod,
    farthest_point_sampling,
    sample_numerical_df,
)


@pytest.mark.parametrize("fraction", [0.2, 0.8, 1.0, 1.2, 2.0, 2.4, 3.5])
@pytest.mark.parametrize("method", list(DiscreteSamplingMethod))
def test_discrete_sampling(fraction, method):
    """Size consistency tests for discrete sampling utility."""
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))

    n_points = math.ceil(fraction * len(df))
    sampled = sample_numerical_df(df, n_points, method=method)

    assert (
        len(sampled) == n_points
    ), "Sampling did not return expected number of points."
    if fraction >= 1.0:
        # Ensure the entire dataframe is contained in the sampled points
        assert (
            pd.merge(df, sampled, how="left", indicator=True)["_merge"].eq("both").all()
        ), "Oversized sampling did not return all original points at least once."
    else:
        # Assure all points are unique
        assert len(sampled) == len(
            sampled.drop_duplicates()
        ), "Undersized sampling did not return unique points."


@given(
    points=hnp.arrays(
        dtype=float,
        shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=1),
        # Because of the square involved in the Euclidean distance computation,
        # we limit the floating point range to avoid overflow problems
        elements=st.floats(min_value=-1e100, max_value=1e100, allow_nan=False),
    )
)
# Explicitly test scenario with equidistant points (see comments in test body)
@example(points=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
def test_farthest_point_sampling(points: np.ndarray):
    """FPS produces the same point sequence regardless of the order in which the
    points are provided. Also, each point fulfills the "farthest point" criterion
    in its respective iteration.
    """  # noqa
    # Order the points using FPS
    sorting_idxs = farthest_point_sampling(points, len(points))
    target = points[sorting_idxs]

    # For the ordered collection of points, it must hold:
    # ---------------------------------------------------
    # The minimum distance of the n_th selected point to all previously selected points
    # must be larger than the minimum distance of any other remaining candidate point to
    # the previously selected points – that's what makes it the "farthest point" in the
    # respective iteration.
    #
    # For the check, we start with the second point (because there are otherwise no
    # previous points) and end with the second last point (because there are otherwise
    # no alternative candidates left):
    dist_mat = pairwise_distances(target)
    for i in range(1, len(dist_mat) - 1):
        min_dist_selected_to_previous = np.min(dist_mat[i, :i])
        min_dist_remaining_to_previous = np.min(dist_mat[i + 1 :, :i])
        z = min_dist_selected_to_previous >= min_dist_remaining_to_previous
        assert z

    # Also, for the algorithm to be fully deterministic, the obtained result should not
    # depend on the particular (random) order in which the points are provided. That is,
    # running the algorithm on a permutation should still produce the same sequence of
    # points. Note: We establish the check on the point coordinates and not the
    # selection index, because the latter can still differ in case of duplicated points.
    #
    # Examples where this can make a difference is three points forming an equilateral
    # triangle or four points spanning a unit cube. Here, tie-breaking operations such
    # as `np.max` can lead to different results depending on the order.
    permutation_idxs = np.random.permutation(len(points))
    sorting_idxs = farthest_point_sampling(points[permutation_idxs], len(points))
    assert np.array_equal(target, points[permutation_idxs][sorting_idxs])

    # Because requesting a single point needs special treatment in FPS,
    # we test this as additional case
    sorting_idxs = farthest_point_sampling(points[permutation_idxs], 1)
    assert np.array_equal(target[[0]], points[permutation_idxs][sorting_idxs])


def test_farthest_point_sampling_pathological_case():
    """FPS executed on a degenerate point cloud raises a warning."""
    points = np.ones((3, 3))
    with pytest.warns(UserWarning, match="identical"):
        selection = farthest_point_sampling(points, 2)
    assert selection == [0, 1]
