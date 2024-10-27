"""Tests for diagnostic utilities."""

import inspect

import pandas as pd
import pytest

import baybe.utils.diagnostics as diag
from baybe._optional.diagnostics import shap
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.searchspace import SearchSpaceType
from baybe.utils.basic import get_subclasses
from tests.conftest import run_iterations


def has_required_init_parameters(cls):
    """Helpfer function checks if initializer has required standard parameters."""
    required_parameters = ["self", "model", "data"]
    init_signature = inspect.signature(cls.__init__)
    parameters = list(init_signature.parameters.keys())
    return parameters[:3] == required_parameters


valid_explainers = [
    getattr(shap.explainers.other, cls_name)
    for cls_name in shap.explainers.other.__all__
    if has_required_init_parameters(getattr(shap.explainers.other, cls_name))
]

valid_hybrid_bayesian_recommenders = [
    TwoPhaseMetaRecommender(recommender=cls())
    for cls in get_subclasses(BayesianRecommender)
    if cls.compatibility == SearchSpaceType.HYBRID
]


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
@pytest.mark.parametrize(
    "parameter_names",
    [["Categorical_1", "SomeSetting", "Num_disc_1", "Conti_finite1"]],
)
def test_shapley_with_measurements(campaign):
    """Test the explain functionalities with measurements."""
    """Test the default explainer in experimental
    and computational representations."""
    run_iterations(campaign, n_iterations=2, batch_size=1)

    for computational_representation in [False, True]:
        shap_val = diag.explanation(
            campaign,
            computational_representation=computational_representation,
        )
        assert isinstance(shap_val, shap.Explanation)

    """Ensure that an error is raised if the data
    to be explained has a different number of parameters."""
    df = pd.DataFrame({"Num_disc_1": [0, 2]})
    with pytest.raises(
        ValueError,
        match=(
            "The provided data does not have the same "
            "amount of parameters as the shap explainer background."
        ),
    ):
        diag.explanation(campaign, data=df)


@pytest.mark.parametrize(
    "parameter_names",
    [["Categorical_1", "SomeSetting", "Num_disc_1", "Conti_finite1"]],
)
def test_non_shapley_explainers(campaign):
    """Test the explain functionalities with the non-SHAP explainer MAPLE."""
    run_iterations(campaign, n_iterations=2, batch_size=1)

    for explainer_cls in valid_explainers:
        try:
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
            other_explainer = diag.explanation(
                campaign,
                computational_representation=True,
                explainer_class=explainer_cls,
            )
            assert isinstance(other_explainer, shap.Explanation)
        except NotImplementedError:
            pass
