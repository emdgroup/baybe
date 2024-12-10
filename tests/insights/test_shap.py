"""Tests for insights subpackage."""

from unittest import mock

import pandas as pd
import pytest
from pytest import param

from baybe._optional.info import INSIGHTS_INSTALLED
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.searchspace import SearchSpaceType
from baybe.utils.basic import get_subclasses
from tests.conftest import run_iterations

pytestmark = pytest.mark.skipif(
    not INSIGHTS_INSTALLED, reason="Optional insights dependency not installed."
)

if INSIGHTS_INSTALLED:
    import shap

    from baybe import insights
    from baybe.insights.shap import SHAPInsight

    default_shap_plots = SHAPInsight.DEFAULT_SHAP_PLOTS
    shap_explainers, non_shap_explainers = (
        SHAPInsight.SHAP_EXPLAINERS,
        SHAPInsight.NON_SHAP_EXPLAINERS,
    )
else:
    default_shap_plots, shap_explainers, non_shap_explainers = [], [], []

valid_hybrid_bayesian_recommenders = [
    param(TwoPhaseMetaRecommender(recommender=cls()), id=f"{cls.__name__}")
    for cls in get_subclasses(BayesianRecommender)
    if cls.compatibility == SearchSpaceType.HYBRID
]


def _test_shap_insights(campaign, explainer_cls, use_comp_rep, is_shap):
    """Helper function for general SHAP explainer tests."""
    run_iterations(campaign, n_iterations=2, batch_size=1)
    try:
        shap_insights = SHAPInsight.from_campaign(
            campaign,
            explainer_class=explainer_cls,
            computational_representation=use_comp_rep,
        )
        assert isinstance(shap_insights, insights.SHAPInsight)
        assert isinstance(
            shap_insights.explainer,
            SHAPInsight.ALL_EXPLAINERS[explainer_cls],
        )
        assert shap_insights._is_shap_explainer == is_shap
        shap_explanation = shap_insights.explanation
        assert isinstance(shap_explanation, shap.Explanation)
        df = pd.DataFrame({"Num_disc_1": [0, 2]})
        with pytest.raises(
            ValueError,
            match="The provided data does not have the same "
            "amount of parameters as the shap explainer background.",
        ):
            shap_insights._init_explanation(df)
    except TypeError as e:
        if (
            "The selected explainer class does not support the campaign surrogate."
            in str(e)
        ):
            pass
        else:
            raise e
    except NotImplementedError as e:
        if (
            "The selected explainer class does not support experimental "
            "representation. Switch to computational representation or "
            "use a different explainer (e.g. the default "
            "shap.KernelExplainer)."
            in str(e)
            and not use_comp_rep
            and not isinstance(explainer_cls, shap.explainers.KernelExplainer)
        ):
            pass
        else:
            raise e


@pytest.mark.slow
@pytest.mark.parametrize("recommender", valid_hybrid_bayesian_recommenders)
@pytest.mark.parametrize("n_grid_points", [5], ids=["grid5"])
@pytest.mark.parametrize("explainer_cls", shap_explainers)
@pytest.mark.parametrize("use_comp_rep", [False, True], ids=["exp", "comp"])
@pytest.mark.parametrize(
    "parameter_names",
    [
        ["Conti_finite1", "Conti_finite2"],
        ["Categorical_1", "SomeSetting", "Num_disc_1", "Conti_finite1"],
    ],
    ids=["continuous_params", "hybrid_params"],
)
def test_shapley_with_measurements(campaign, explainer_cls, use_comp_rep):
    """Test the explain functionalities with measurements."""
    _test_shap_insights(campaign, explainer_cls, use_comp_rep, is_shap=True)


@pytest.mark.parametrize("explainer_cls", non_shap_explainers)
@pytest.mark.parametrize("n_grid_points", [5], ids=["grid5"])
@pytest.mark.parametrize(
    "parameter_names",
    [["Categorical_1", "SomeSetting", "Num_disc_1", "Conti_finite1"]],
    ids=["hybrid_params"],
)
def test_non_shapley_explainers(campaign, explainer_cls):
    """Test the explain functionalities with the non-SHAP explainer MAPLE."""
    """Test the non-SHAP explainer in computational representation."""
    _test_shap_insights(campaign, explainer_cls, use_comp_rep=True, is_shap=False)


@pytest.mark.slow
@pytest.mark.parametrize("recommender", valid_hybrid_bayesian_recommenders)
@pytest.mark.parametrize("n_grid_points", [5], ids=["grid5"])
@pytest.mark.parametrize("use_comp_rep", [False, True], ids=["exp", "comp"])
@pytest.mark.parametrize("plot_type", default_shap_plots)
@pytest.mark.parametrize(
    "parameter_names",
    [["Categorical_1", "SomeSetting", "Num_disc_1", "Conti_finite1"]],
    ids=["hybrid_params"],
)
def test_shap_insight_plots(campaign, use_comp_rep, plot_type):
    """Test the default SHAP plots."""
    run_iterations(campaign, n_iterations=2, batch_size=1)
    shap_insights = SHAPInsight.from_campaign(
        campaign,
        computational_representation=use_comp_rep,
    )
    with mock.patch("matplotlib.pyplot.show"):
        shap_insights.plot(plot_type)


@pytest.mark.parametrize("recommender", valid_hybrid_bayesian_recommenders)
@pytest.mark.parametrize("n_grid_points", [5], ids=["grid5"])
def test_updated_campaign_explanations(campaign):
    """Test explanations for campaigns with updated measurements."""
    with pytest.raises(
        ValueError,
        match="The campaign does not contain any measurements.",
    ):
        shap_insights = SHAPInsight.from_campaign(campaign)
    run_iterations(campaign, n_iterations=2, batch_size=1)
    shap_insights = SHAPInsight.from_campaign(campaign)
    explanation_two_iter = shap_insights.explanation
    run_iterations(campaign, n_iterations=2, batch_size=1)
    shap_insights = SHAPInsight.from_campaign(campaign)
    explanation_four_iter = shap_insights.explanation
    assert explanation_two_iter != explanation_four_iter


@pytest.mark.parametrize("recommender", valid_hybrid_bayesian_recommenders)
@pytest.mark.parametrize("n_grid_points", [5], ids=["grid5"])
def test_shap_insights_from_recommender(campaign):
    """Test the creation of SHAP insights from a recommender."""
    run_iterations(campaign, n_iterations=2, batch_size=1)
    recommender = campaign.recommender.recommender
    shap_insight = SHAPInsight.from_recommender(
        recommender,
        campaign.searchspace,
        campaign.objective,
        campaign.measurements,
    )
    assert isinstance(shap_insight, insights.SHAPInsight)
