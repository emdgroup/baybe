"""Tests for diagnostic utilities."""

import pandas as pd
import pytest
import shap

import baybe.utils.diagnostics as diag
from baybe import Campaign
from baybe.objective import SingleTargetObjective
from baybe.parameters import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget


@pytest.fixture
def diagnostics_campaign():
    """Create a campaign with a hybrid space including substances."""
    parameters = [
        NumericalDiscreteParameter("NumDisc", values=(0, 1, 2)),
        NumericalContinuousParameter("NumCont", bounds=(2, 3)),
        SubstanceParameter(
            name="Molecules",
            data={
                "TAP": "C12=CC=CC=C1N=C3C(C=C(N=C(C=CC=C4)C4=N5)C5=C3)=N2",
                "Pyrene": "C1(C=CC2)=C(C2=CC=C3CC=C4)C3=C4C=C1",
            },
            encoding="MORDRED",
        ),
    ]
    searchspace = SearchSpace.from_product(parameters=parameters)
    target = NumericalTarget(name="y_1", mode="MAX")
    objective = SingleTargetObjective(target=target)
    campaign = Campaign(searchspace, objective)
    return campaign


@pytest.fixture
def diagnostics_campaign_activated(diagnostics_campaign):
    """Create an activated campaign with a hybrid space including substances.

    Measurements were added and first recommendations were made.
    """
    diagnostics_campaign.add_measurements(
        pd.DataFrame(
            {
                "NumDisc": [0, 2],
                "NumCont": [2.2, 2.8],
                "Molecules": ["Pyrene", "TAP"],
                "y_1": [0.5, 0.7],
            }
        )
    )
    diagnostics_campaign.recommend(3)
    return diagnostics_campaign


def test_shapley_values_no_measurements(diagnostics_campaign):
    """A campaign without measurements raises an error."""
    with pytest.raises(ValueError, match="No measurements have been provided yet."):
        diag.explanation(diagnostics_campaign)


def test_shapley_with_measurements(diagnostics_campaign_activated):
    """Test the explain functionalities with measurements."""
    """Test the default explainer in experimental representation."""
    shap_val = diag.explanation(diagnostics_campaign_activated)
    assert isinstance(shap_val, shap.Explanation)

    """Test the default explainer in computational representation."""
    shap_val_comp = diag.explanation(
        diagnostics_campaign_activated,
        computational_representation=True,
    )
    assert isinstance(shap_val_comp, shap.Explanation)

    """Ensure that an error is raised if the data
    to be explained has a different number of parameters."""
    df = pd.DataFrame(
        {
            "NumDisc": [0, 2],
            "NumCont": [2.2, 2.8],
            "Molecules": ["Pyrene", "TAP"],
            "ExtraParam": [0, 1],
        }
    )
    with pytest.raises(
        ValueError,
        match=(
            "The provided data does not have the same "
            "amount of parameters as the shap explainer background."
        ),
    ):
        diag.explanation(diagnostics_campaign_activated, data=df)


def test_non_shapley_explainers(diagnostics_campaign_activated):
    """Test the explain functionalities with the non-SHAP explainer MAPLE."""
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
            diagnostics_campaign_activated,
            computational_representation=False,
            explainer_class=shap.explainers.other.Maple,
        )

    """Test the MAPLE explainer in computational representation."""
    maple_explainer = diag.explainer(
        diagnostics_campaign_activated,
        computational_representation=True,
        explainer_class=shap.explainers.other.Maple,
    )
    assert isinstance(maple_explainer, shap.explainers.other._maple.Maple)
