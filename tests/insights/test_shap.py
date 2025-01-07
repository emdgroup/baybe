"""Tests for insights subpackage."""

from unittest import mock

import pandas as pd
import pytest
from pytest import mark

from baybe._optional.info import SHAP_INSTALLED
from baybe.campaign import Campaign
from tests.conftest import run_iterations

# File-wide parameterization settings
pytestmark = [
    mark.skipif(not SHAP_INSTALLED, reason="Optional shap package not installed."),
    mark.parametrize("n_grid_points", [5], ids=["g5"]),
    mark.parametrize("n_iterations", [2], ids=["i2"]),
    mark.parametrize("batch_size", [2], ids=["b2"]),
    mark.parametrize(
        "parameter_names",
        [
            ["Conti_finite1", "Conti_finite2"],
            ["Categorical_1", "SomeSetting", "Num_disc_1", "Conti_finite1"],
        ],
        ids=["conti_params", "hybrid_params"],
    ),
]


if SHAP_INSTALLED:
    from baybe import insights
    from baybe._optional.insights import shap
    from baybe.insights.shap import (
        ALL_EXPLAINERS,
        NON_SHAP_EXPLAINERS,
        SHAP_EXPLAINERS,
        SUPPORTED_SHAP_PLOTS,
        SHAPInsight,
    )
else:
    ALL_EXPLAINERS = []
    NON_SHAP_EXPLAINERS = []
    SHAP_EXPLAINERS = []
    SUPPORTED_SHAP_PLOTS = []


def _test_shap_insight(campaign, explainer_cls, use_comp_rep, is_shap):
    """Helper function for general SHAP explainer tests."""
    # run_iterations(campaign, n_iterations=2, batch_size=5)
    try:
        # Sanity check explainer
        shap_insight = SHAPInsight.from_campaign(
            campaign,
            explainer_cls=explainer_cls,
            use_comp_rep=use_comp_rep,
        )
        assert isinstance(shap_insight, insights.SHAPInsight)
        assert isinstance(
            shap_insight.explainer,
            ALL_EXPLAINERS[explainer_cls],
        )
        assert shap_insight.uses_shap_explainer == is_shap

        # Sanity check explanation
        df = campaign.measurements[[p.name for p in campaign.parameters]]
        if use_comp_rep:
            df = campaign.searchspace.transform(df)
        shap_explanation = shap_insight.explain(df)
        assert isinstance(shap_explanation, shap.Explanation)
    except TypeError as e:
        if "The selected explainer class" in str(e):
            pytest.xfail("Unsupported model/explainer combination")
        else:
            raise e
    except NotImplementedError as e:
        if (
            "The selected explainer class" in str(e)
            and not use_comp_rep
            and not isinstance(explainer_cls, shap.explainers.KernelExplainer)
        ):
            pytest.xfail("Exp. rep. not supported")
        else:
            raise e


@mark.slow
@mark.parametrize("explainer_cls", SHAP_EXPLAINERS)
@mark.parametrize("use_comp_rep", [False, True], ids=["exp", "comp"])
def test_shap_explainers(ongoing_campaign, explainer_cls, use_comp_rep):
    """Test the explain functionalities with measurements."""
    _test_shap_insight(ongoing_campaign, explainer_cls, use_comp_rep, is_shap=True)


@mark.parametrize("explainer_cls", NON_SHAP_EXPLAINERS)
def test_non_shap_explainers(ongoing_campaign, explainer_cls):
    """Test the explain functionalities with the non-SHAP explainer MAPLE."""
    """Test the non-SHAP explainer in computational representation."""
    _test_shap_insight(
        ongoing_campaign, explainer_cls, use_comp_rep=True, is_shap=False
    )


@mark.slow
@mark.parametrize("explainer_cls", ["KernelExplainer"], ids=["KernelExplainer"])
@mark.parametrize("use_comp_rep", [False, True], ids=["exp", "comp"])
def test_invalid_explained_data(ongoing_campaign, explainer_cls, use_comp_rep):
    """Test invalid explained data."""
    shap_insight = SHAPInsight.from_campaign(
        ongoing_campaign,
        explainer_cls=explainer_cls,
        use_comp_rep=use_comp_rep,
    )
    df = pd.DataFrame({"Num_disc_1": [0, 2]})
    with pytest.raises(
        ValueError,
        match="The provided data does not have the same amount of parameters as the "
        "shap explainer background.",
    ):
        shap_insight.explain(df)


@mark.slow
@mark.parametrize("use_comp_rep", [False, True], ids=["exp", "comp"])
@mark.parametrize("plot_type", SUPPORTED_SHAP_PLOTS)
def test_plots(ongoing_campaign: Campaign, use_comp_rep, plot_type):
    """Test the default SHAP plots."""
    shap_insight = SHAPInsight.from_campaign(
        ongoing_campaign,
        use_comp_rep=use_comp_rep,
    )
    df = ongoing_campaign.measurements[[p.name for p in ongoing_campaign.parameters]]
    if use_comp_rep:
        df = ongoing_campaign.searchspace.transform(df)
    with mock.patch("matplotlib.pyplot.show"):
        shap_insight.plot(df, plot_type=plot_type)


def test_updated_campaign_explanations(campaign, n_iterations, batch_size):
    """Test explanations for campaigns with updated measurements."""
    with pytest.raises(
        ValueError,
        match="The campaign does not contain any measurements.",
    ):
        SHAPInsight.from_campaign(campaign)

    run_iterations(campaign, n_iterations=n_iterations, batch_size=batch_size)
    shap_insight = SHAPInsight.from_campaign(campaign)
    df = campaign.measurements[[p.name for p in campaign.parameters]]
    explanation_1 = shap_insight.explain(df)

    run_iterations(campaign, n_iterations=n_iterations, batch_size=batch_size)
    shap_insight = SHAPInsight.from_campaign(campaign)
    df = campaign.measurements[[p.name for p in campaign.parameters]]
    explanation_2 = shap_insight.explain(df)

    assert explanation_1 != explanation_2, "SHAP explanations should not be identical."


def test_creation_from_recommender(ongoing_campaign):
    """Test the creation of SHAP insights from a recommender."""
    recommender = ongoing_campaign.recommender.recommender
    shap_insight = SHAPInsight.from_recommender(
        recommender,
        ongoing_campaign.searchspace,
        ongoing_campaign.objective,
        ongoing_campaign.measurements,
    )
    assert isinstance(shap_insight, insights.SHAPInsight)
