"""Tests features of the Campaign object."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest import param

from baybe.campaign import Campaign
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.searchspace.discrete import SubspaceDiscrete

from .conftest import run_iterations


@pytest.mark.parametrize(
    "target_names",
    [
        param(["Target_max"], id="max"),
        param(["Target_min"], id="min"),
        param(["Target_max_bounded"], id="max_b"),
        param(["Target_min_bounded"], id="min_b"),
        param(["Target_match_bell"], id="match_bell"),
        param(["Target_match_triangular"], id="match_tri"),
        param(
            ["Target_max_bounded", "Target_min_bounded", "Target_match_triangular"],
            id="desirability",
        ),
    ],
)
@pytest.mark.parametrize("batch_size", [2], ids=["b2"])
@pytest.mark.parametrize("n_iterations", [2], ids=["i2"])
def test_get_surrogate(campaign, n_iterations, batch_size):
    """Test successful extraction of the surrogate model."""
    run_iterations(campaign, n_iterations, batch_size)

    model = campaign.get_surrogate()
    assert model is not None, "Something went wrong during surrogate model extraction."


@pytest.mark.parametrize(
    ("anti", "expected"),
    [
        (
            False,
            pd.DataFrame(columns=["a", "b"], data=[[0.0, 3.0], [0.0, 4.0], [0.0, 5.0]]),
        ),
        (
            True,
            pd.DataFrame(columns=["a", "b"], data=[[1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]),
        ),
    ],
    ids=["regular", "anti"],
)
def test_candidate_filter(anti, expected):
    """The candidate filter extracts the correct subset of points."""
    subspace = SubspaceDiscrete.from_product(
        [
            NumericalDiscreteParameter("a", [0, 1]),
            NumericalDiscreteParameter("b", [3, 4, 5]),
        ]
    )
    campaign = Campaign(subspace)
    df = campaign.toggle_discrete_candidates(pd.DataFrame({"a": [0]}), False, anti=anti)
    assert_frame_equal(
        df, pd.merge(df.reset_index(), expected).set_index("index"), check_names=False
    )
