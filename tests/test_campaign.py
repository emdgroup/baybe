"""Tests features of the Campaign object."""

from contextlib import nullcontext
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from attrs import evolve
from pandas.testing import assert_index_equal
from pytest import param

from baybe.acquisition import qLogEI, qLogNEHVI, qTS, qUCB
from baybe.campaign import _EXCLUDED, Campaign
from baybe.constraints.conditions import SubSelectionCondition
from baybe.constraints.discrete import DiscreteExcludeConstraint
from baybe.exceptions import NotEnoughPointsLeftError
from baybe.objectives import DesirabilityObjective, ParetoObjective
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.recommenders.base import RecommenderProtocol
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.pure.nonpredictive.sampling import (
    FPSRecommender,
    RandomRecommender,
)
from baybe.searchspace.core import SearchSpaceType
from baybe.searchspace.discrete import SubspaceDiscrete
from baybe.surrogates import (
    BetaBernoulliMultiArmedBanditSurrogate,
    GaussianProcessSurrogate,
)
from baybe.targets import BinaryTarget, NumericalTarget
from baybe.utils.basic import UNSPECIFIED
from baybe.utils.dataframe import add_fake_measurements
from tests.conftest import run_iterations


@pytest.fixture
def campaign_for_flag_test(request) -> Campaign:
    """A mocked campaign for testing the allow_* flags."""
    searchspace = NumericalDiscreteParameter("p", [0, 1]).to_searchspace()
    objective = NumericalTarget("t").to_objective()
    flags = {
        "allow_recommending_already_measured": True,
        "allow_recommending_already_recommended": True,
        "allow_recommending_pending_experiments": True,
        **request.param,
    }
    campaign = Campaign(searchspace, objective, **flags)
    return evolve(
        campaign,
        recommender=Mock(wraps=TwoPhaseMetaRecommender(), spec=RecommenderProtocol),
    )


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
    "campaign_for_flag_test",
    [{"allow_recommending_already_measured": v} for v in [True, False]],
    ids=["True", "False"],
    indirect=True,
)
def test_allow_measured_flag(campaign_for_flag_test: Campaign):
    """The flag controls the candidate set and properly interacts with the cache."""
    campaign = campaign_for_flag_test
    flag = campaign.allow_recommending_already_measured
    mock_recommend = campaign.recommender.recommend

    assert campaign._cached_recommendation is None

    for i in range(3):
        with (
            nullcontext() if i < 2 or flag else pytest.raises(NotEnoughPointsLeftError)
        ):
            # Because the data context is different in each loop iteration,
            # the recommender must be called instead of using the cache
            rec = campaign.recommend(1)
            assert mock_recommend.call_count == i + 1

            # A follow-up call should use the cache
            campaign.recommend(1)
            assert mock_recommend.call_count == i + 1

        add_fake_measurements(rec, campaign.targets)
        campaign.add_measurements(rec)


@pytest.mark.parametrize(
    "campaign_for_flag_test",
    [{"allow_recommending_already_recommended": v} for v in [True, False]],
    ids=["True", "False"],
    indirect=True,
)
def test_allow_recommended_flag(campaign_for_flag_test: Campaign):
    """The flag controls the candidate set and properly interacts with the cache."""
    campaign = campaign_for_flag_test
    flag = campaign.allow_recommending_already_recommended
    mock_recommend = campaign.recommender.recommend

    # Pre-populate the cache
    assert campaign._cached_recommendation is None
    rec = campaign.recommend(1)
    mock_recommend.reset_mock()
    assert campaign._cached_recommendation.equals(rec)

    for i in range(2):
        with (
            nullcontext() if i < 1 or flag else pytest.raises(NotEnoughPointsLeftError)
        ):
            # Depending on the flag, we always reuse the cache or always recompute
            campaign.recommend(1)
            assert mock_recommend.call_count == 0 if flag else i + 1


@pytest.mark.parametrize(
    "campaign_for_flag_test",
    [{"allow_recommending_pending_experiments": v} for v in [True, False]],
    ids=["True", "False"],
    indirect=True,
)
def test_allow_pending_flag(campaign_for_flag_test: Campaign):
    campaign = campaign_for_flag_test
    flag = campaign.allow_recommending_pending_experiments
    mock_recommend = campaign.recommender.recommend
    pending = pd.DataFrame({"p": [0]})

    # We add some data so that we are in the model-based regime
    campaign.add_measurements(pd.DataFrame({"p": [0], "t": [0]}))

    # Pre-populate the cache
    assert campaign._cached_recommendation is None
    rec = campaign.recommend(1)
    mock_recommend.reset_mock()
    assert campaign._cached_recommendation.equals(rec)

    # Recommending without pending experiments uses the cache
    campaign.recommend(1)
    mock_recommend.assert_not_called()

    # With pending experiments, the recommender is called again
    campaign.recommend(1, pending_experiments=pending)
    mock_recommend.assert_called_once()

    # Changing the pending context once again, retriggers computation once more
    campaign.recommend(1)
    assert mock_recommend.call_count == 2

    # Depending on the flag, we can run out of candidates
    with nullcontext() if flag else pytest.raises(NotEnoughPointsLeftError):
        campaign.recommend(1, pending_experiments=pd.DataFrame({"p": [0, 1]}))
        assert mock_recommend.call_count == 3


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
    assert ongoing_campaign._cached_recommendation is None


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
                    NumericalTarget.normalized_ramp("t1", cutoffs=(0, 1)),
                    NumericalTarget.normalized_ramp(
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
@pytest.mark.parametrize("n_iterations", [1], ids=["i1"])
@pytest.mark.parametrize("batch_size", [1], ids=["b3"])
def test_posterior_stats_invalid_input(ongoing_campaign, stats, error, match):
    """Invalid inputs for posterior statistics raise expected exceptions."""
    with pytest.raises(error, match=match):
        ongoing_campaign.posterior_stats(ongoing_campaign.measurements, stats)


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


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        param("recommender", RandomRecommender(), id="recommender"),
        param("allow_recommending_already_measured", True, id="measured"),
        param("allow_recommending_already_recommended", False, id="recommended"),
        param("allow_recommending_pending_experiments", False, id="pending"),
    ],
)
@pytest.mark.parametrize("change", [True, False], ids=["change", "no_change"])
def test_cache_invalidation(
    campaign: Campaign, attribute: str, value: Any, change: bool
):
    """Altering mutable public attributes invalidates the cache."""
    if isinstance(value, bool):
        new_value = value ^ change
    else:
        # Important: Even if we do not change, we use a new instance of the same class
        #   to test that equality is a sufficient condition for not clearing the cache
        new_value = FPSRecommender() if change else RandomRecommender()

    setattr(campaign, attribute, value)
    campaign._cache_recommendation(pd.DataFrame())
    assert campaign._cached_recommendation is not None
    setattr(campaign, attribute, new_value)
    assert getattr(campaign, attribute) is new_value
    if change:
        assert campaign._cached_recommendation is None
    else:
        assert campaign._cached_recommendation is not None
