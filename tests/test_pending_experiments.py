"""Tests pending experiments mechanism."""

import warnings

import numpy as np
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
from baybe.searchspace.core import SearchSpaceType
from baybe.utils.basic import get_subclasses
from baybe.utils.dataframe import add_parameter_noise
from baybe.utils.random import temporary_seed

_discrete_params = ["Categorical_1", "Switch_1", "Num_disc_1"]
_continuous_params = ["Conti_finite1", "Conti_finite2", "Conti_finite3"]
_hybrid_params = ["Categorical_1", "Num_disc_1", "Conti_finite1", "Conti_finite2"]


@pytest.mark.parametrize(
    "parameter_names, recommender",
    [
        param(
            _discrete_params,
            FPSRecommender(),
            id="fps_discrete",
        ),
        param(
            _discrete_params,
            PAMClusteringRecommender(),
            id="pam_discrete",
        ),
        param(
            _discrete_params,
            KMeansClusteringRecommender(),
            id="kmeans_discrete",
        ),
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
        param(
            _discrete_params,
            TwoPhaseMetaRecommender(recommender=BotorchRecommender()),
            id="botorch_discrete_allow",
        ),
        param(
            _continuous_params,
            TwoPhaseMetaRecommender(recommender=BotorchRecommender()),
            id="botorch_continuous_allow",
        ),
        param(
            _hybrid_params,
            TwoPhaseMetaRecommender(recommender=BotorchRecommender()),
            id="botorch_hybrid_allow",
        ),
        param(
            _discrete_params,
            NaiveHybridSpaceRecommender(
                disc_recommender=FPSRecommender(),
            ),
            id="naive1_discrete",
        ),
        param(
            _discrete_params,
            NaiveHybridSpaceRecommender(
                disc_recommender=KMeansClusteringRecommender(),
            ),
            id="naive2_discrete",
        ),
    ],
)
def test_pending_points(campaign, batch_size, fake_measurements):
    """Test there is no recommendation overlap if pending experiments are specified."""
    warnings.filterwarnings("ignore", category=UnusedObjectWarning)

    # Repeated recommendations explicitly need to be allowed or the potential overlap
    # will be avoided trivially
    if campaign.searchspace.type == SearchSpaceType.DISCRETE:
        campaign.allow_recommending_already_recommended = True
        campaign.allow_recommending_already_measured = True

    # Add some initial measurements
    campaign.add_measurements(fake_measurements)

    # Get recommendations and set them as pending experiments while getting another set
    # Fix the random seed for each recommend call to limit influence of randomness in
    # some recommenders which could also trivially avoid overlap
    with temporary_seed(1337):
        rec1 = campaign.recommend(batch_size)
    campaign.clear_cache()
    with temporary_seed(1337):
        rec2 = campaign.recommend(batch_size=batch_size, pending_experiments=rec1)

    # Assert they have no overlap, round to avoid numerical fluctuation
    overlap = pd.merge(rec1.round(3), rec2.round(3), how="inner")
    assert len(overlap) == 0, (
        f"Recommendations are overlapping!\n\nRecommendations 1:\n{rec1}\n\n"
        f"Recommendations 2:\n{rec2}\n\nOverlap:\n{overlap}"
    )


acqfs_non_pending = [
    a()
    for a in get_subclasses(AcquisitionFunction)
    if not a.supports_pending_experiments
]


@pytest.mark.parametrize(
    "acqf", acqfs_non_pending, ids=[a.abbreviation for a in acqfs_non_pending]
)
@pytest.mark.parametrize(
    "parameter_names",
    [
        param(_discrete_params, id="discrete"),
        param(_continuous_params, id="continuous"),
        param(_hybrid_params, id="hybrid"),
    ],
)
@pytest.mark.parametrize("batch_size", [1], ids=["b1"])
def test_invalid_acqf(searchspace, objective, batch_size, acqf, fake_measurements):
    """Test exception raised for acqfs that don't support pending experiments."""
    recommender = TwoPhaseMetaRecommender(
        recommender=BotorchRecommender(acquisition_function=acqf)
    )

    # Create fake measurements and pending experiments
    fake_pending_experiments = fake_measurements.copy()
    add_parameter_noise(fake_pending_experiments, searchspace.parameters)

    with pytest.raises(IncompatibleAcquisitionFunctionError):
        recommender.recommend(
            batch_size,
            searchspace,
            objective,
            measurements=fake_measurements,
            pending_experiments=fake_pending_experiments,
        )


@pytest.mark.parametrize(
    "parameter_names, invalid_pending_value",
    [
        param(["Categorical_1", "Num_disc_1"], "asd", id="cat_param_invalid_value"),
        param(["Categorical_1", "Num_disc_1"], 1337, id="cat_param_num"),
        param(["Categorical_1", "Num_disc_1"], np.nan, id="cat_param_nan"),
        param(["Num_disc_1", "Num_disc_2"], "asd", id="num_param_str"),
        param(["Num_disc_1", "Num_disc_2"], np.nan, id="num_param_nan"),
        param(["Custom_1", "Num_disc_2"], "asd", id="custom_param_str"),
        param(["Custom_1", "Num_disc_2"], 1337, id="custom_param_num"),
        param(["Custom_1", "Num_disc_2"], np.nan, id="custom_param_nan"),
        param(["Task", "Num_disc_1"], "asd", id="task_param_invalid_value"),
        param(["Task", "Num_disc_1"], 1337, id="task_param_num"),
        param(["Task", "Num_disc_1"], np.nan, id="task_param_nan"),
    ],
)
@pytest.mark.parametrize("batch_size", [3], ids=["b3"])
def test_invalid_input(
    searchspace,
    recommender,
    objective,
    batch_size,
    invalid_pending_value,
    parameter_names,
    fake_measurements,
):
    """Test exception raised for invalid pending experiments input."""
    # Create fake measurements and pending experiments
    fake_pending_experiments = fake_measurements.copy()
    fake_pending_experiments[parameter_names[0]] = invalid_pending_value

    with pytest.raises((ValueError, TypeError), match="parameter"):
        recommender.recommend(
            batch_size,
            searchspace,
            objective,
            measurements=fake_measurements,
            pending_experiments=fake_pending_experiments,
        )
