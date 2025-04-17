"""Tests features of the Campaign object."""

from contextlib import nullcontext

import pandas as pd
import pytest
from pandas.testing import assert_index_equal
from pytest import param

from baybe.acquisition import qLogEI, qLogNEHVI, qTS, qUCB
from baybe.campaign import _EXCLUDED, Campaign
from baybe.constraints.conditions import SubSelectionCondition
from baybe.constraints.discrete import DiscreteExcludeConstraint
from baybe.objectives import DesirabilityObjective, ParetoObjective
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace.core import SearchSpaceType
from baybe.searchspace.discrete import SubspaceDiscrete
from baybe.surrogates import (
    BetaBernoulliMultiArmedBanditSurrogate,
    GaussianProcessSurrogate,
)
from baybe.targets import BinaryTarget, NumericalTarget
from baybe.utils.basic import UNSPECIFIED

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


@pytest.mark.parametrize("complement", [False, True], ids=["regular", "complement"])
@pytest.mark.parametrize("exclude", [True, False], ids=["exclude", "include"])
@pytest.mark.parametrize(
    "constraints",
    [
        pd.DataFrame({"a": [0]}),
        [DiscreteExcludeConstraint(["a"], [SubSelectionCondition([1])])],
    ],
    ids=["dataframe", "constraints"],
)
def test_candidate_toggling(constraints, exclude, complement):
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
    campaign.toggle_discrete_candidates(constraints, exclude, complement=complement)

    # Extract row indices of candidates whose metadata should have been toggled
    matches = campaign.searchspace.discrete.exp_rep["a"] == 0
    idx = matches.index[~matches if complement else matches]

    # Assert that metadata is set correctly
    target = campaign._searchspace_metadata.loc[idx, _EXCLUDED]
    other = campaign._searchspace_metadata[_EXCLUDED].drop(index=idx)
    assert all(target == exclude)  # must contain the updated values
    assert all(other != exclude)  # must contain the original values


@pytest.mark.parametrize(
    "flag",
    [
        "allow_recommending_already_measured",
        "allow_recommending_already_recommended",
        "allow_recommending_pending_experiments",
    ],
    ids=lambda x: x.removeprefix("allow_recommending_"),
)
@pytest.mark.parametrize(
    "space_type",
    [SearchSpaceType.DISCRETE, SearchSpaceType.CONTINUOUS],
    ids=lambda x: x.name,
)
@pytest.mark.parametrize(
    "value", [True, False, param(UNSPECIFIED, id=repr(UNSPECIFIED))]
)
def test_setting_allow_flags(flag, space_type, value):
    """Passed allow_* flags are rejected if incompatible with the search space type."""
    kwargs = {flag: value}
    expect_error = (space_type is SearchSpaceType.DISCRETE) != (
        value is not UNSPECIFIED
    )

    if space_type is SearchSpaceType.DISCRETE:
        parameter = NumericalDiscreteParameter("p", [0, 1])
    else:
        parameter = NumericalContinuousParameter("p", [0, 1])

    with pytest.raises(ValueError) if expect_error else nullcontext():
        Campaign(parameter, **kwargs)


@pytest.mark.parametrize(
    "parameter_names", [["Categorical_1", "Categorical_2", "Num_disc_1"]]
)
@pytest.mark.parametrize("n_iterations", [3], ids=["i3"])
def test_update_measurements(ongoing_campaign):
    """Updating measurements makes the expected changes."""
    p_name, t_name = "Num_disc_1", ongoing_campaign.targets[0].name
    updated = ongoing_campaign.measurements.iloc[[0], :]

    # Perform the update
    updated.iloc[0, updated.columns.get_loc(p_name)] = 1337
    updated.iloc[0, updated.columns.get_loc(t_name)] = 1337
    ongoing_campaign.update_measurements(
        updated, numerical_measurements_must_be_within_tolerance=False
    )

    # Make sure values are updated and resets have been made
    meas = ongoing_campaign.measurements
    assert meas.iloc[0, updated.columns.get_loc(p_name)] == 1337
    assert meas.iloc[0, updated.columns.get_loc(t_name)] == 1337
    assert meas.iloc[[0], updated.columns.get_loc("FitNr")].isna().all()
    assert ongoing_campaign._cached_recommendation.empty


@pytest.mark.parametrize(
    ("parameter_names", "objective", "surrogate_model", "acqf", "batch_size"),
    [
        param(
            ["Categorical_1", "Num_disc_1", "Conti_finite1"],
            NumericalTarget("t1").to_objective(),
            GaussianProcessSurrogate(),
            qLogEI(),
            3,
            id="single_target",
        ),
        param(
            ["Categorical_1", "Num_disc_1", "Conti_finite1"],
            DesirabilityObjective(
                (
                    NumericalTarget.clamped_affine("t1", cutoffs=(0, 1)),
                    NumericalTarget.clamped_affine(
                        "t2", cutoffs=(0, 1), descending=True
                    ),
                )
            ),
            GaussianProcessSurrogate(),
            qLogEI(),
            3,
            id="desirability",
        ),
        param(
            ["Categorical_1", "Num_disc_1", "Conti_finite1"],
            ParetoObjective(
                (NumericalTarget("t1"), NumericalTarget("t2", minimize=True))
            ),
            GaussianProcessSurrogate(),
            qLogNEHVI(),
            3,
            id="pareto",
        ),
        param(
            ["Categorical_1"],
            BinaryTarget(name="Target_binary").to_objective(),
            BetaBernoulliMultiArmedBanditSurrogate(),
            qTS(),
            1,
            id="bernoulli",
        ),
    ],
)
@pytest.mark.parametrize("n_grid_points", [5], ids=["g5"])
@pytest.mark.parametrize("n_iterations", [1], ids=["i1"])
def test_posterior_stats(ongoing_campaign, n_iterations, batch_size):
    """Posterior statistics have expected shape, index and columns."""
    objective = ongoing_campaign.objective
    tested_stats = ["mean", "std", "var"]
    test_quantiles = not isinstance(objective.targets[0], BinaryTarget)
    if test_quantiles:
        tested_stats += [0.05, 0.95]

    stats = ongoing_campaign.posterior_stats(
        ongoing_campaign.measurements, tested_stats
    )

    # Assert number of entries and index
    (
        assert_index_equal(ongoing_campaign.measurements.index, stats.index),
        (ongoing_campaign.measurements.index, stats.index),
    )

    # Assert expected columns are present.
    match objective:
        case DesirabilityObjective():
            targets = ["Desirability"]
        case _:
            targets = [t.name for t in objective.targets]

    for t in targets:
        for stat in tested_stats:
            stat_name = f"Q_{stat}" if isinstance(stat, float) else stat
            assert sum(f"{t}_{stat_name}" in x for x in stats.columns) == 1, (
                f"{t}_{stat_name} not in the returned posterior statistics"
            )

    # Assert no NaN's present
    assert not stats.isna().any().any()

    if not test_quantiles:
        # Assert correct error for unsupported quantile calculation
        with pytest.raises(
            TypeError, match="does not support the statistic associated"
        ):
            ongoing_campaign.posterior_stats(ongoing_campaign.measurements, [0.1])


@pytest.mark.parametrize(
    ("stats", "error", "match"),
    [
        param(
            ["invalid"],
            TypeError,
            "does not support the statistic associated",
            id="invalid_stat",
        ),
        param(
            [-0.1],
            ValueError,
            "quantile statistics can only be",
            id="quantile_too_small",
        ),
        param(
            [1.1],
            ValueError,
            "quantile statistics can only be",
            id="quantile_too_large",
        ),
    ],
)
@pytest.mark.parametrize("n_grid_points", [5], ids=["g5"])
@pytest.mark.parametrize("n_iterations", [1], ids=["i1"])
@pytest.mark.parametrize("batch_size", [1], ids=["b3"])
def test_posterior_stats_invalid_input(ongoing_campaign, stats, error, match):
    """Invalid inputs for posterior statistics raise expected exceptions."""
    with pytest.raises(error, match=match):
        ongoing_campaign.posterior_stats(ongoing_campaign.measurements, stats)


@pytest.mark.parametrize("n_grid_points", [5], ids=["g5"])
@pytest.mark.parametrize("n_iterations", [1], ids=["i1"])
@pytest.mark.parametrize("batch_size", [3], ids=["b3"])
def test_acquisition_value_computation(ongoing_campaign: Campaign):
    """Acquisition values have the expected shape."""
    df = ongoing_campaign.searchspace.discrete.exp_rep
    assert not df.empty

    # Using campaign acquisition function
    acqfs = ongoing_campaign.acquisition_values(df)
    assert_index_equal(acqfs.index, df.index)
    joint_acqf = ongoing_campaign.joint_acquisition_value(df)
    assert isinstance(joint_acqf, float)

    # Using separate acquisition function
    acqfs = ongoing_campaign.acquisition_values(df, qUCB())
    assert_index_equal(acqfs.index, df.index)
    joint_acqf = ongoing_campaign.joint_acquisition_value(df, qUCB())
    assert isinstance(joint_acqf, float)
