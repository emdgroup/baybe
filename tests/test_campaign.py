"""Tests features of the Campaign object."""

import pandas as pd
import pytest
from pytest import param

from baybe.campaign import _EXCLUDED, Campaign
from baybe.constraints.conditions import SubSelectionCondition
from baybe.constraints.discrete import DiscreteExcludeConstraint
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


@pytest.mark.parametrize("anti", [False, True], ids=["regular", "anti"])
@pytest.mark.parametrize("exclude", [True, False], ids=["exclude", "include"])
@pytest.mark.parametrize(
    "constraint",
    [
        pd.DataFrame({"a": [0]}),
        DiscreteExcludeConstraint(["a"], [SubSelectionCondition([1])]),
    ],
    ids=["dataframe", "constraints"],
)
def test_candidate_toggling(constraint, exclude, anti):
    """Toggling discrete candidates updates the campaign metadata accordingly."""
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
    campaign.toggle_discrete_candidates(constraint, exclude, anti=anti)

    # Extract row indices of candidates whose metadata should have been toggled
    matches = campaign.searchspace.discrete.exp_rep["a"] == 0
    idx = matches.index[~matches] if anti else matches.index[matches]

    # Assert that metadata is set correctly
    target = campaign._searchspace_metadata.loc[idx, _EXCLUDED]
    other = campaign._searchspace_metadata[_EXCLUDED].drop(index=idx)
    assert all(target == exclude)  # must contain the updated values
    assert all(other != exclude)  # must contain the original values
