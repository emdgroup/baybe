"""Tests for the lookup and imputation functionality."""

from contextlib import nullcontext

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from pytest import param

from baybe.simulation.lookup import look_up_targets
from baybe.targets import NumericalTarget


@pytest.fixture
def lookup():
    """Create a sample lookup dataframe."""
    return pd.DataFrame(
        {
            "p": [1, 2, 3, 4, 5],
            "t_max": [10, 20, 30, 40, 50],
            "t_min": [10, 20, 30, 40, 50],
            "t_match": [10, 20, 30, 40, 50],
        }
    )


@pytest.fixture
def query():
    """Create an imputation query."""
    return pd.DataFrame({"p": [0]})


@pytest.fixture
def targets():
    """Create example targets."""
    return [
        NumericalTarget("t_max"),
        NumericalTarget("t_min", minimize=True),
        NumericalTarget.match_triangular("t_match", 40, cutoffs=(10, 60)),
    ]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        param("mean", {"t_max": 30, "t_min": 30, "t_match": 30}, id="mean"),
        param("worst", {"t_max": 10, "t_min": 50, "t_match": 10}, id="worst"),
        param("best", {"t_max": 50, "t_min": 10, "t_match": 40}, id="best"),
    ],
)
def test_target_imputation(query, targets, lookup, mode, expected):
    """Test the target imputation function."""
    look_up_targets(query, targets, lookup, mode)
    assert_frame_equal(
        query[[t.name for t in targets]],
        pd.DataFrame(expected, index=[0]),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    ("query", "match_iloc"),
    [
        param(pd.DataFrame({"p": [2]}), 1, id="exact_match"),
        param(pd.DataFrame({"p": [0]}), None, id="no_match"),
    ],
)
def test_target_imputation_exact(query, targets, lookup, match_iloc):
    """Test the target imputation function for exact matches."""
    with pytest.raises(IndexError) if match_iloc is None else nullcontext():
        look_up_targets(query, targets, lookup, impute_mode="error")

    if match_iloc is not None:
        cols = [t.name for t in targets]
        expected = lookup[cols].iloc[match_iloc]
        assert_series_equal(query[cols].iloc[0], expected, check_names=False)


def test_target_imputation_random(targets, lookup):
    """Test the target imputation function for random selection."""
    query = pd.DataFrame({"p": range(-5, 0)})
    assert pd.merge(query, lookup).empty

    look_up_targets(query, targets, lookup, impute_mode="random")

    cols = [t.name for t in targets]
    merged = pd.merge(lookup, query, on=cols)
    assert len(merged) == len(query)
