"""Tests pending points mechanism."""

import warnings

import pandas as pd
import pytest
from pytest import param

from baybe.acquisition.base import AcquisitionFunction
from baybe.exceptions import IncompatibleAcquisitionFunctionError, UnusedObjectWarning
from baybe.recommenders import (
    BotorchRecommender,
    FPSRecommender,
    GaussianMixtureClusteringRecommender,
    KMeansClusteringRecommender,
    PAMClusteringRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.utils.basic import get_subclasses
from baybe.utils.dataframe import add_fake_results, add_parameter_noise

_discrete_params = ["Categorical_1", "Switch_1", "Num_disc_1"]
_continuous_params = ["Conti_finite1", "Conti_finite2", "Conti_finite3"]
_hybrid_params = ["Categorical_1", "Num_disc_1", "Conti_finite1", "Conti_finite2"]


@pytest.mark.parametrize(
    "parameter_names, recommender",
    [
        param(_discrete_params, FPSRecommender(), id="fps_discrete"),
        param(_discrete_params, PAMClusteringRecommender(), id="pam_discrete"),
        param(_discrete_params, KMeansClusteringRecommender(), id="kmeans_discrete"),
        param(
            _discrete_params,
            GaussianMixtureClusteringRecommender(),
            id="gm_discrete",
        ),
        param(
            _discrete_params,
            TwoPhaseMetaRecommender(recommender=BotorchRecommender()),
            id="botorch_discrete",
        ),
        param(
            _continuous_params,
            TwoPhaseMetaRecommender(recommender=BotorchRecommender()),
            id="botorch_continuous",
        ),
        param(
            _hybrid_params,
            TwoPhaseMetaRecommender(recommender=BotorchRecommender()),
            id="botorch_hybrid",
        ),
    ],
)
def test_pending_points(campaign, batch_size):
    """Test there is no recommendation overlap if pending points are specified."""
    warnings.filterwarnings("ignore", category=UnusedObjectWarning)

    # Perform a fake first iteration
    rec = campaign.recommend(batch_size)
    add_fake_results(rec, campaign.targets)
    campaign.add_measurements(rec)

    # Get recommendations and set them as pending while getting another set
    rec1 = campaign.recommend(batch_size)
    campaign._cached_recommendation = pd.DataFrame()  # ensure no recommendation cache
    rec2 = campaign.recommend(batch_size=batch_size, pending_measurements=rec1)

    # Assert they have no overlap, round to avoid numerical fluctuation
    overlap = pd.merge(rec1.round(3), rec2.round(3), how="inner")
    assert len(overlap) == 0, (
        f"Recommendations are overlapping!\n\nRecommendations 1:\n{rec1}\n\n"
        f"Recommendations 2:\n{rec2}\n\nOverlap:\n{overlap}"
    )


_non_mc_acqfs = [a() for a in get_subclasses(AcquisitionFunction) if not a.is_mc]


@pytest.mark.parametrize(
    "acqf", _non_mc_acqfs, ids=[a.abbreviation for a in _non_mc_acqfs]
)
@pytest.mark.parametrize(
    "parameter_names",
    [
        param(_discrete_params, id="discrete"),
        param(_continuous_params, id="continuous"),
        param(_hybrid_params, id="hybrid"),
    ],
)
@pytest.mark.parametrize("n_grid_points", [5], ids=["g5"])
@pytest.mark.parametrize("batch_size", [3], ids=["b3"])
def test_invalid_acqf(searchspace, recommender, objective, batch_size, acqf):
    """Test exception raised for acqfs that don't support pending points."""
    recommender = TwoPhaseMetaRecommender(
        recommender=BotorchRecommender(acquisition_function=acqf)
    )

    # Get recommendation and add a fake results
    rec1 = recommender.recommend(batch_size, searchspace, objective)
    add_fake_results(rec1, objective.targets)

    # Create fake pending measurements
    rec2 = rec1.copy()
    add_parameter_noise(rec2, searchspace.parameters)

    with pytest.raises(IncompatibleAcquisitionFunctionError):
        recommender.recommend(
            batch_size,
            searchspace,
            objective,
            measurements=rec1,
            pending_measurements=rec2,
        )
