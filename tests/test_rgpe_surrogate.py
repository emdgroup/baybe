"""Tests for the rank-weighted Gaussian process ensemble (RGPE) surrogate."""

from typing import Literal

import pandas as pd
import pytest
import torch

from baybe import Campaign
from baybe.exceptions import (
    IncompatibleSearchSpaceError,
    IncompatibleSurrogateError,
)
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter, TaskParameter
from baybe.parameters.enum import TransferLearningMode
from baybe.recommenders import BotorchRecommender, TwoPhaseMetaRecommender
from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate, RGPESurrogate
from baybe.targets import NumericalTarget

_TARGET = "A"
_SOURCE = "B"


def _make_searchspace(
    *, override: bool = False, values=(_TARGET, _SOURCE), active=(_TARGET,)
) -> SearchSpace:
    """Build a single-task-parameter transfer-learning search space."""
    mode = TransferLearningMode.RGPE if override else None
    return SearchSpace.from_product(
        [
            NumericalContinuousParameter("x", (0, 5)),
            TaskParameter(
                "task",
                values=values,
                active_values=active,
                override_transfer_learning_mode=mode,
            ),
        ]
    )


def _make_measurements(
    training_data: Literal["source", "target", "both"] = "both",
) -> pd.DataFrame:
    """Build measurements for the source and/or target task.

    Uses a fixed lookup rather than the random fake-data utilities because the ranking
    weights need controlled values and exact per-task point counts.
    """
    lookup = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5],
            "y": [1.0, 2.0, 3.0, 4.0, 1.2, 2.1, 2.9, 4.2],
            "task": [_TARGET] * 4 + [_SOURCE] * 4,
        }
    )
    if training_data == "source":
        return lookup[lookup["task"] == _SOURCE].reset_index(drop=True)
    if training_data == "target":
        return lookup[lookup["task"] == _TARGET].reset_index(drop=True)
    return lookup


@pytest.fixture(name="objective")
def fixture_objective() -> SingleTargetObjective:
    """A single-target maximization objective."""
    return NumericalTarget("y").to_objective()


@pytest.fixture(name="candidates")
def fixture_candidates() -> pd.DataFrame:
    """Candidate points in experimental representation (target task)."""
    return pd.DataFrame({"x": [0.75, 2.25, 4.5], "task": [_TARGET] * 3})


def test_fit_and_posterior(objective, candidates):
    """The fitted ensemble yields a valid posterior with normalized weights."""
    surrogate = RGPESurrogate(n_mc_samples=32)
    surrogate.fit(_make_searchspace(), objective, _make_measurements("both"))

    # One source GP, one target GP, and normalized weights over both.
    assert len(surrogate._source_gps) == 1
    assert surrogate._target_gp is not None
    assert surrogate._weights.shape == (2,)
    assert torch.all(surrogate._weights >= 0)
    assert surrogate._weights.sum().item() == pytest.approx(1.0)

    posterior = surrogate.posterior(candidates)
    assert posterior.mean.numel() == len(candidates)
    assert torch.all(posterior.variance > 0)
    assert torch.isfinite(posterior.mean).all()


def test_inner_gps_use_identity_mode_and_do_not_redispatch(objective):
    """Inner GPs fit on the identity-mode space instead of re-dispatching to RGPE.

    The inner GPs run on the identity-mode space, whose task parameter no longer carries
    the RGPE mode. This is what prevents infinite recursion in the factory dispatch.
    """
    surrogate = RGPESurrogate(n_mc_samples=32)
    surrogate.fit(_make_searchspace(), objective, _make_measurements("both"))

    for gp in (*surrogate._source_gps, surrogate._target_gp):
        assert gp is not None
        # A concrete GP was fitted rather than dispatched to another RGPE ensemble.
        assert gp._delegate is None
        assert gp._model is not None
        # The inner space carries the identity mode, which is the recursion guard.
        task_param = gp._searchspace._task_parameter
        assert task_param is not None
        assert (
            task_param.override_transfer_learning_mode is TransferLearningMode.IDENTITY
        )


def test_override_dispatch_matches_direct(objective, candidates):
    """Selecting RGPE via the task override dispatches to an RGPE delegate."""
    measurements = _make_measurements("both")

    gp = GaussianProcessSurrogate()
    gp.fit(_make_searchspace(override=True), objective, measurements)
    assert isinstance(gp._delegate, RGPESurrogate)
    # The undelegated GP model is never built along the dispatch path.
    assert gp._model is None

    # The dispatched delegate produces the same posterior as a direct RGPE surrogate
    # (up to Monte Carlo noise in the weights).
    direct = RGPESurrogate()
    direct.fit(_make_searchspace(), objective, measurements)

    delegated_mean = gp.posterior(candidates).mean
    direct_mean = direct.posterior(candidates).mean
    assert torch.allclose(delegated_mean, direct_mean, atol=0.1)


@pytest.mark.parametrize(
    ("measurements", "target_fitted", "expected_weights"),
    [
        pytest.param(
            _make_measurements("source"),
            False,
            [1.0],
            id="cold-start",
        ),
        pytest.param(
            pd.concat(
                [
                    _make_measurements("source"),
                    pd.DataFrame({"x": [2.0], "y": [2.0], "task": [_TARGET]}),
                ],
                ignore_index=True,
            ),
            True,
            [0.5, 0.5],
            id="single-target-point",
        ),
    ],
)
def test_insufficient_target_data_uses_uniform_weights(
    objective, candidates, measurements, target_fitted, expected_weights
):
    """Without enough target points to rank, the ensemble weights all models uniformly.

    This covers both the cold start (no target data, averaging over the source models)
    and the single-target-point case (too few points for the ranking loss).
    """
    surrogate = RGPESurrogate(n_mc_samples=32)
    surrogate.fit(_make_searchspace(), objective, measurements)

    assert (surrogate._target_gp is not None) == target_fitted
    assert torch.allclose(surrogate._weights, torch.tensor(expected_weights))
    assert torch.isfinite(surrogate.posterior(candidates).mean).all()


@pytest.mark.parametrize(
    ("searchspace", "training_data", "match"),
    [
        pytest.param(
            SearchSpace.from_product([NumericalContinuousParameter("x", (0, 5))]),
            "both",
            "task parameter",
            id="no-task-parameter",
        ),
        pytest.param(
            _make_searchspace(active=(_TARGET, _SOURCE)),
            "both",
            "exactly one active",
            id="multiple-active-tasks",
        ),
        pytest.param(
            _make_searchspace(),
            "target",
            "source task",
            id="missing-source-measurements",
        ),
    ],
)
def test_invalid_transfer_learning_setup_is_rejected(
    searchspace, training_data, match, objective
):
    """RGPE rejects an invalid task configuration or missing source measurements."""
    with pytest.raises(IncompatibleSearchSpaceError, match=match):
        RGPESurrogate().fit(searchspace, objective, _make_measurements(training_data))


@pytest.mark.parametrize("via_override", [False, True], ids=["direct", "override"])
def test_campaign_recommendation(objective, via_override):
    """A campaign recommends target-task points using the RGPE surrogate."""
    searchspace = _make_searchspace(override=via_override)
    surrogate = (
        GaussianProcessSurrogate() if via_override else RGPESurrogate(n_mc_samples=32)
    )
    recommender = TwoPhaseMetaRecommender(
        recommender=BotorchRecommender(surrogate_model=surrogate)
    )
    campaign = Campaign(searchspace, objective, recommender)
    campaign.add_measurements(_make_measurements("both"))

    recommendations = campaign.recommend(batch_size=2)
    assert (recommendations["task"] == _TARGET).all()


def test_posterior_mean_function_rejects_delegate(objective):
    """A dispatched GP has no single mean module and fails loudly if asked for one."""
    gp = GaussianProcessSurrogate()
    gp.fit(_make_searchspace(override=True), objective, _make_measurements("both"))
    with pytest.raises(IncompatibleSurrogateError, match="not implemented"):
        gp.posterior_mean_function(
            _make_searchspace(), objective, _make_measurements("both")
        )
