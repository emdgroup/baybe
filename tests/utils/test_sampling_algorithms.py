"""Tests for sampling algorithm utilities."""
import math

import numpy as np
import pandas as pd
import pytest

from baybe.utils.sampling_algorithms import DiscreteSamplingMethod, sample_numerical_df


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
