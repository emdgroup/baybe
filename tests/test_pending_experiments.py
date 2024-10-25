"""Tests pending experiments mechanism."""

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
    NaiveHybridSpaceRecommender,
    PAMClusteringRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.utils.basic import get_subclasses
from baybe.utils.dataframe import add_fake_measurements, add_parameter_noise
from baybe.utils.random import temporary_seed

_discrete_params = ["Categorical_1", "Switch_1", "Num_disc_1"]
_continuous_params = ["Conti_finite1", "Conti_finite2", "Conti_finite3"]
_hybrid_params = ["Categorical_1", "Num_disc_1", "Conti_finite1", "Conti_finite2"]

# Repeated recommendations explicitly need to be allowed or the potential overlap will
# be avoided trivially
_flags = dict(
    allow_repeated_recommendations=True,
    allow_recommending_already_measured=True,
)


@pytest.mark.parametrize(
    "parameter_names, recommender",
    [
        param(
            _discrete_params,
            FPSRecommender(**_flags),
            id="fps_discrete",
        ),
        param(_discrete_params, PAMClusteringRecommender(**_flags), id="pam_discrete"),
        param(
            _discrete_params,
            KMeansClusteringRecommender(**_flags),
            id="kmeans_discrete",
        ),
        param(
            _discrete_params,
            GaussianMixtureClusteringRecommender(**_flags),
            id="gm_discrete",
        ),
        param(
            _discrete_params,
            TwoPhaseMetaRecommender(recommender=BotorchRecommender(**_flags)),
            id="botorch_discrete",
        ),
        param(
            _continuous_params,
            TwoPhaseMetaRecommender(recommender=BotorchRecommender(**_flags)),
            id="botorch_continuous",
        ),
        param(
            _hybrid_params,
            TwoPhaseMetaRecommender(recommender=BotorchRecommender(**_flags)),
            id="botorch_hybrid",
        ),
        param(
            _discrete_params,
            TwoPhaseMetaRecommender(
                recommender=BotorchRecommender(
                    **_flags, allow_recommending_pending_experiments=True
                )
            ),
            id="botorch_discrete_allow",
        ),
        param(
            _continuous_params,
            TwoPhaseMetaRecommender(
                recommender=BotorchRecommender(
                    **_flags, allow_recommending_pending_experiments=True
                )
            ),
            id="botorch_continuous_allow",
        ),
        param(
            _hybrid_params,
            TwoPhaseMetaRecommender(
                recommender=BotorchRecommender(
                    **_flags, allow_recommending_pending_experiments=True
                )
            ),
            id="botorch_hybrid_allow",
        ),
        param(
            _discrete_params,
            NaiveHybridSpaceRecommender(
                disc_recommender=FPSRecommender(**_flags), **_flags
            ),
            id="naive1_discrete",
        ),
        param(
            _discrete_params,
            NaiveHybridSpaceRecommender(
                disc_recommender=KMeansClusteringRecommender(**_flags), **_flags
            ),
            id="naive2_discrete",
        ),
    ],
)
@pytest.mark.parametrize("n_grid_points", [8], ids=["grid8"])
def test_pending_points(campaign, batch_size):
    """Test there is no recommendation overlap if pending experiments are specified."""
    warnings.filterwarnings("ignore", category=UnusedObjectWarning)

    # Perform a fake first iteration
    rec = campaign.recommend(batch_size)
    add_fake_measurements(rec, campaign.targets)
    campaign.add_measurements(rec)

    # Get recommendations and set them as pending experiments while getting another set
    # Fix the random seed for each recommend call to limit influence of randomness in
    # some recommenders which could also trivially avoid overlap
    with temporary_seed(1337):
        rec1 = campaign.recommend(batch_size)
    campaign._cached_recommendation = pd.DataFrame()  # ensure no recommendation cache
    with temporary_seed(1337):
        rec2 = campaign.recommend(batch_size=batch_size, pending_experiments=rec1)

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
    """Test exception raised for acqfs that don't support pending experiments."""
    recommender = TwoPhaseMetaRecommender(
        recommender=BotorchRecommender(acquisition_function=acqf)
    )

    # Get recommendation and add a fake results
    rec1 = recommender.recommend(batch_size, searchspace, objective)
    add_fake_measurements(rec1, objective.targets)

    # Create fake pending experiments
    rec2 = rec1.copy()
    add_parameter_noise(rec2, searchspace.parameters)

    with pytest.raises(IncompatibleAcquisitionFunctionError):
        recommender.recommend(
            batch_size,
            searchspace,
            objective,
            measurements=rec1,
            pending_experiments=rec2,
        )
