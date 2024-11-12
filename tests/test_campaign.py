"""Tests features of the Campaign object."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest import param

from baybe.campaign import _EXCLUDED, Campaign
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
@pytest.mark.parametrize("exclude", [True, False], ids=["exclude", "include"])
def test_candidate_filter(exclude, anti, expected):
    """The candidate filter extracts the correct subset of points and the campaign
    metadata is updated accordingly."""  # noqa

    subspace = SubspaceDiscrete.from_product(
        [
            NumericalDiscreteParameter("a", [0, 1]),
            NumericalDiscreteParameter("b", [3, 4, 5]),
        ]
    )
    campaign = Campaign(subspace)

    # Set metadata to opposite of targeted value so that we can verify the effect later
    campaign._searchspace_metadata[_EXCLUDED] = not exclude

    # Toggle the candidates
    df = campaign.toggle_discrete_candidates(
        pd.DataFrame({"a": [0]}), exclude, anti=anti
    )

    # Assert that the filtering is correct
    rows = pd.merge(df.reset_index(), expected).set_index("index")
    assert_frame_equal(df, rows, check_names=False)

    # Assert that metadata is set correctly
    target = campaign._searchspace_metadata.loc[rows.index, _EXCLUDED]
    other = campaign._searchspace_metadata[_EXCLUDED].drop(index=rows.index)
    assert all(target == exclude)  # must contain the updated values
    assert all(other != exclude)  # must contain the original values
