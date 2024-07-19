"""Tests pending points mechanism."""

import pandas as pd
import pytest
from pytest import param

from baybe.recommenders import (
    BotorchRecommender,
    FPSRecommender,
    GaussianMixtureClusteringRecommender,
    KMeansClusteringRecommender,
    PAMClusteringRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.utils.dataframe import add_fake_results

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
    # Perform a fake first iteration
    rec = campaign.recommend(batch_size)
    add_fake_results(rec, campaign.targets)
    campaign.add_measurements(rec)

    # Get recommendations, set them as pending and get another set of recommendations
    rec1 = campaign.recommend(batch_size)
    rec2 = campaign.recommend(batch_size=batch_size, pending_measurements=rec1)

    # Assert they have no overlap
    overlap = pd.merge(rec1, rec2, how="inner")
    assert len(overlap) == 0, (
        f"Recommendations are overlapping!\n\nRecommendations 1:\n{rec1}\n\n"
        f"Recommendations 2:\n{rec2}\n\nOverlap:\n{overlap}"
    )
