"""Tests for diagnostic utilities."""

import inspect

import pandas as pd
import pytest
from pytest import param

from baybe._optional.info import DIAGNOSTICS_INSTALLED
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.searchspace import SearchSpaceType
from baybe.utils.basic import get_subclasses
from tests.conftest import run_iterations

pytestmark = pytest.mark.skipif(
    not DIAGNOSTICS_INSTALLED, reason="Optional diagnostics dependency not installed."
)

if DIAGNOSTICS_INSTALLED:
    import shap

    from baybe import diagnostics as diag

EXCLUDED_EXPLAINER_KEYWORDS = ["Tree", "GPU", "Gradient", "Sampling", "Deep"]


def _has_required_init_parameters(cls):
    """Helper function checks if initializer has required standard parameters."""
    REQUIRED_PARAMETERS = ["self", "model", "data"]
    init_signature = inspect.signature(cls.__init__)
    parameters = list(init_signature.parameters.keys())
    return parameters[:3] == REQUIRED_PARAMETERS


non_shap_explainers = (
    [
        param(explainer, id=f"{cls_name}")
        for cls_name in shap.explainers.other.__all__
        if _has_required_init_parameters(
            explainer := getattr(shap.explainers.other, cls_name)
        )
        and all(x not in cls_name for x in EXCLUDED_EXPLAINER_KEYWORDS)
    ]
    if DIAGNOSTICS_INSTALLED
    else []
)

shap_explainers = (
    [
        param(getattr(shap.explainers, cls_name), id=f"{cls_name}")
        for cls_name in shap.explainers.__all__
        if all(x not in cls_name for x in EXCLUDED_EXPLAINER_KEYWORDS)
    ]
    if DIAGNOSTICS_INSTALLED
    else []
)

valid_hybrid_bayesian_recommenders = [
    param(TwoPhaseMetaRecommender(recommender=cls()), id=f"{cls.__name__}")
    for cls in get_subclasses(BayesianRecommender)
    if cls.compatibility == SearchSpaceType.HYBRID
]


def _test_explainer(campaign, explainer_cls, use_comp_rep):
    """Helper function for general explainer tests."""
    run_iterations(campaign, n_iterations=2, batch_size=1)
    try:
        shap_val = diag.explanation(
            campaign,
            computational_representation=use_comp_rep,
            explainer_class=explainer_cls,
        )
        assert isinstance(shap_val, shap.Explanation)
        df = pd.DataFrame({"Num_disc_1": [0, 2]})
        with pytest.raises(
            ValueError,
            match="The provided data does not have the same "
            "amount of parameters as specified for the campaign.",
        ):
            diag.explanation(
                campaign,
                data=df,
                computational_representation=True,
                explainer_class=explainer_cls,
            )
    except TypeError as e:
        if (
            "The selected explainer class does not support the campaign surrogate."
            in str(e)
        ):
            pass
        else:
            raise e


@pytest.mark.parametrize(
    "parameter_names",
    [["Categorical_1", "SomeSetting", "Num_disc_1", "Conti_finite1"]],
)
def test_shapley_values_no_measurements(campaign):
    """A campaign without measurements raises an error."""
    with pytest.raises(ValueError, match="No measurements have been provided yet."):
        diag.explanation(campaign)


@pytest.mark.slow
@pytest.mark.parametrize("recommender", valid_hybrid_bayesian_recommenders)
@pytest.mark.parametrize("explainer_cls", shap_explainers)
@pytest.mark.parametrize("use_comp_rep", [False, True], ids=["exp", "comp"])
@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
def test_shapley_with_measurements_continuous(campaign, explainer_cls, use_comp_rep):
    """Test the explain functionalities with measurements."""
    _test_explainer(campaign, explainer_cls, use_comp_rep)


@pytest.mark.slow
@pytest.mark.parametrize("recommender", valid_hybrid_bayesian_recommenders)
@pytest.mark.parametrize("explainer_cls", shap_explainers)
@pytest.mark.parametrize("use_comp_rep", [False, True], ids=["exp", "comp"])
@pytest.mark.parametrize(
    "parameter_names",
    [["Categorical_1", "SomeSetting", "Num_disc_1", "Conti_finite1"]],
)
def test_shapley_with_measurements(campaign, explainer_cls, use_comp_rep):
    """Test the explain functionalities with measurements."""
    _test_explainer(campaign, explainer_cls, use_comp_rep)


@pytest.mark.parametrize("explainer_cls", non_shap_explainers)
@pytest.mark.parametrize(
    "parameter_names",
    [["Categorical_1", "SomeSetting", "Num_disc_1", "Conti_finite1"]],
    ids=["params1"],
)
def test_non_shapley_explainers(campaign, explainer_cls):
    """Test the explain functionalities with the non-SHAP explainer MAPLE."""
    """Test the non-SHAP explainer in computational representation."""
    _test_explainer(campaign, explainer_cls, use_comp_rep=True)
    """Ensure that an error is raised if non-computational representation
        is used with a non-Kernel SHAP explainer."""
    with pytest.raises(
        ValueError,
        match=(
            "Experimental representation is not supported "
            "for non-Kernel SHAP explainer."
        ),
    ):
        diag.explanation(
            campaign,
            computational_representation=False,
            explainer_class=explainer_cls,
        )
