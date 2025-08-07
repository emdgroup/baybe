"""Tests for sampling algorithm utilities."""

import math
from unittest.mock import patch

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given
from pytest import param
from sklearn.metrics import pairwise_distances

from baybe.recommenders.pure.nonpredictive.sampling import (
    FPSAMPLE_USED,
    FPSInitialization,
    FPSRecommender,
)
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

    assert len(sampled) == n_points, (
        "Sampling did not return expected number of points."
    )
    if fraction >= 1.0:
        # Ensure the entire dataframe is contained in the sampled points
        assert (
            pd.merge(df, sampled, how="left", indicator=True)["_merge"].eq("both").all()
        ), "Oversized sampling did not return all original points at least once."
    else:
        # Assure all points are unique
        assert len(sampled) == len(sampled.drop_duplicates()), (
            "Undersized sampling did not return unique points."
        )


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
@pytest.mark.parametrize("random_tie_break", [False, True])
@example(points=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
def test_farthest_point_sampling(points: np.ndarray, random_tie_break: bool):
    """FPS produces the same point sequence regardless of the order in which the
    points are provided. Also, each point fulfills the "farthest point" criterion
    in its respective iteration.
    """  # noqa
    # Order the points using FPS
    sorting_idxs = farthest_point_sampling(
        points, len(points), random_tie_break=random_tie_break
    )
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

    # Also, if the algorithm is set to fully deterministic, the obtained result should
    # not depend on the particular (random) order in which the points are provided.
    # That is, running the algorithm on a permutation should still produce the same
    # sequence of points. The flag `random_tie_break` can adjust the deterministic
    # behaviour. Note: We establish the check on the point coordinates and not the
    # selection index, because the latter can still differ in case of duplicated points.
    #
    # Examples where this can make a difference is three points forming an equilateral
    # triangle or four points spanning a unit cube. Here, tie-breaking operations such
    # as `np.max` can lead to different results depending on the order.
    permutation_idxs = np.random.permutation(len(points))
    sorting_idxs = farthest_point_sampling(
        points[permutation_idxs], len(points), random_tie_break=random_tie_break
    )
    if not random_tie_break:
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


@pytest.mark.skipif(
    not FPSAMPLE_USED,
    reason="Optional 'fpsample' package not used.",
)
@pytest.mark.parametrize(
    ("init", "tie_break", "match"),
    [
        param(
            FPSInitialization.RANDOM,
            True,
            "does not support 'FPSInitialization.RANDOM'",
            id="raise_initial_random",
        ),
        param(
            FPSInitialization.FARTHEST,
            True,
            "does not support random tie-breaking",
            id="raise_tiebreak",
        ),
    ],
)
def test_fps_recommender_expected_errors(init, tie_break, match):
    """Test that FPSRecommender emits exceptions for unsupported arguments."""
    with pytest.raises(ValueError, match=match):
        FPSRecommender(initialization=init, random_tie_break=tie_break)


_valid_points = np.array([[1, 1], [2, 2], [3, 3]])


@pytest.mark.parametrize(
    ("points", "n_requested", "initialization", "match"),
    [
        param(
            _valid_points,
            0,
            "farthest",
            "must be at least 1. Provided: n_samples=0.",
            id="n_samples_zero",
        ),
        param(
            _valid_points,
            -1,
            "farthest",
            "must be at least 1. Provided: n_samples=-1.",
            id="n_samples_neg",
        ),
        param(
            np.array([1, 2, 3]),
            2,
            "farthest",
            "must be two-dimensional",
            id="input_not_2D",
        ),
        param(
            _valid_points,
            4,
            "farthest",
            r"number of requested samples \(4\)",
            id="too_many_requested",
        ),
        param(
            _valid_points,
            3,
            [0, 1, 2, 3],
            r"number of provided initialization indices \(4\)",
            id="too_many_initial",
        ),
        param(
            _valid_points,
            3,
            [0, 0],
            r"but contains duplicates: \{0\}",
            id="duplicated_init_points",
        ),
        param(
            _valid_points,
            3,
            [0, -1, 5],
            r"contains out-of-bounds indices: \[-1, 5\]",
            id="invalid_init_points",
        ),
        param(
            _valid_points,
            3,
            "bla",
            "Unknown initialization type.",
            id="unknown_init",
        ),
    ],
)
def test_fps_utility_expected_errors(points, n_requested, initialization, match):
    """FPSRecommender emits exceptions for unsupported arguments."""
    with pytest.raises(ValueError, match=match):
        farthest_point_sampling(points, n_requested, initialization=initialization)


def test_fps_recommender_utility_call(searchspace):
    """FPSRecommender calls expected underlying utility."""
    if FPSAMPLE_USED:
        context = patch("baybe._optional.fpsample.fps_sampling", return_value=[0, 1, 2])
    else:
        context = patch(
            "baybe.recommenders.pure.nonpredictive.sampling.farthest_point_sampling",
            return_value=[0, 1, 2],
        )

    with context as mock_:
        result = FPSRecommender().recommend(batch_size=3, searchspace=searchspace)

    mock_.assert_called_once()
    assert result.index.tolist() == [0, 1, 2]


@pytest.mark.skipif(
    not FPSAMPLE_USED,
    reason="Optional 'fpsample' package not used.",
)
def test_fps_recommender_result_consistency(searchspace):
    """FPS utilities return consistent results."""
    from baybe._optional.fpsample import fps_sampling

    points = searchspace.discrete.comp_rep.values
    inds1 = fps_sampling(points, 3, start_idx=0).tolist()
    inds2 = farthest_point_sampling(
        points, 3, initialization=[0], random_tie_break=False
    )

    assert inds1 == inds2, (inds1, inds2)
