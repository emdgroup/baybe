"""Tests for diagnostic utilities."""

import inspect

import pandas as pd
import pytest

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

    def _has_required_init_parameters(cls):
        """Helper function checks if initializer has required standard parameters."""
        required_parameters = ["model", "data"]
        init_signature = inspect.signature(cls.__init__)
        parameters = list(init_signature.parameters.keys())
        return parameters[:3] == required_parameters

    valid_non_shap_explainers = [
        getattr(shap.explainers.other, cls_name)
        for cls_name in shap.explainers.other.__all__
        if _has_required_init_parameters(getattr(shap.explainers.other, cls_name))
    ]

    shap_explainers = [
        getattr(shap.explainers, cls_name) for cls_name in shap.explainers.__all__
    ]

valid_hybrid_bayesian_recommenders = [
    TwoPhaseMetaRecommender(recommender=cls())
    for cls in get_subclasses(BayesianRecommender)
    if cls.compatibility == SearchSpaceType.HYBRID
]


def _run_explainer_tests(campaign, explainers, representation_types):
    """Helper to test explainers for different representation types."""
    for explainer_cls in explainers:
        for representation in representation_types:
            shap_val = diag.explanation(
                campaign,
                computational_representation=representation,
                explainer_class=explainer_cls,
            )
            assert isinstance(shap_val, shap.Explanation)


def _test_shap_explainers(campaign, explainers, check_param_count=False):
    run_iterations(campaign, n_iterations=2, batch_size=1)
    try:
        _run_explainer_tests(campaign, explainers, [False, True])
        if check_param_count:
            df = pd.DataFrame({"Num_disc_1": [0, 2]})
            with pytest.raises(
                ValueError,
                match="The provided data does not have the same "
                "amount of parameters as specified for the campaign.",
            ):
                diag.explanation(campaign, data=df, explainer_class=shap_explainers[0])
    except ModuleNotFoundError as e:
        if "No module named 'tensorflow'" in str(e):
            pass
        else:
            raise e
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
@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
def test_shapley_with_measurements_continuous(campaign):
    """Test the explain functionalities with measurements."""
    run_iterations(campaign, n_iterations=2, batch_size=1)
    for explainer_cls in shap_explainers:
        _test_shap_explainers(campaign, shap_explainers)


@pytest.mark.slow
@pytest.mark.parametrize("recommender", valid_hybrid_bayesian_recommenders)
@pytest.mark.parametrize(
    "parameter_names",
    [["Categorical_1", "SomeSetting", "Num_disc_1", "Conti_finite1"]],
)
def test_shapley_with_measurements(campaign):
    """Test the explain functionalities with measurements."""
    run_iterations(campaign, n_iterations=2, batch_size=1)
    for explainer_cls in shap_explainers:
        _test_shap_explainers(campaign, shap_explainers)


@pytest.mark.parametrize(
    "parameter_names",
    [["Categorical_1", "SomeSetting", "Num_disc_1", "Conti_finite1"]],
)
def test_non_shapley_explainers(campaign):
    """Test the explain functionalities with the non-SHAP explainer MAPLE."""
    run_iterations(campaign, n_iterations=2, batch_size=1)

    for explainer_cls in valid_non_shap_explainers:
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

        """Test the non-SHAP explainer in computational representation."""
        _run_explainer_tests(campaign, [explainer_cls], [True])
