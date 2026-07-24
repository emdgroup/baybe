"""Tests for the transfer-learning prototype dispatching (Case A + MEAN_TRANSFER)."""

import numpy as np
import pandas as pd
import pytest
import torch

from baybe.acquisition.acqfs import qNegIntegratedPosteriorVariance
from baybe.campaign import Campaign
from baybe.exceptions import (
    IncompatibleAcquisitionFunctionError,
    IncompatibleObjectiveError,
    IncompatibleSearchSpaceError,
    IncompatibleSurrogateError,
)
from baybe.kernels.basic import IndexKernel, MaternKernel, PositiveIndexKernel
from baybe.objectives.desirability import DesirabilityObjective
from baybe.parameters.categorical import TaskParameter
from baybe.parameters.enum import TransferLearningMode
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.presets.baybe import _BayBETaskKernelFactory
from baybe.surrogates.transfer_learning.mean_transfer import MeanTransferSurrogate
from baybe.surrogates.transfer_learning.residual_transfer import (
    ResidualTransferSurrogate,
)
from baybe.surrogates.transfer_learning.rgpe_transfer import RGPETransferSurrogate
from baybe.targets.numerical import NumericalTarget
from baybe.utils.dataframe import add_fake_measurements


@pytest.fixture(name="objective")
def fixture_objective():
    """A single numerical target objective."""
    return NumericalTarget("t").to_objective()


def _make_task_searchspace(values, active_values, mode=None):
    """Build a 1D continuous + task search space with an optional TL override."""
    return SearchSpace.from_product(
        [
            NumericalContinuousParameter("p", (0.0, 1.0)),
            TaskParameter(
                "task",
                values=values,
                active_values=active_values,
                override_transfer_learning_mode=mode,
            ),
        ]
    )


def _make_measurements(values, objective, n_per_task=8):
    """Create fake measurements with ``n_per_task`` rows for each task value."""
    frames = [
        pd.DataFrame({"p": np.linspace(0.0, 1.0, n_per_task), "task": value})
        for value in values
    ]
    measurements = pd.concat(frames, ignore_index=True)
    add_fake_measurements(measurements, objective.targets)
    return measurements


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        pytest.param(None, PositiveIndexKernel, id="default"),
        pytest.param(
            TransferLearningMode.POSITIVE_INDEX_KERNEL,
            PositiveIndexKernel,
            id="positive",
        ),
        pytest.param(TransferLearningMode.INDEX_KERNEL, IndexKernel, id="index"),
    ],
)
def test_task_kernel_dispatch(mode, expected, objective):
    """The task-kernel factory selects the kernel dictated by the override mode."""
    searchspace = _make_task_searchspace(["source", "target"], ["target"], mode)
    measurements = _make_measurements(["source", "target"], objective)

    kernel = _BayBETaskKernelFactory()._make(searchspace, objective, measurements)

    # Exact type check because `PositiveIndexKernel` is a subclass of `IndexKernel`.
    assert type(kernel) is expected


def test_override_with_custom_kernel_raises(objective):
    """Combining an override with a custom kernel raises an error."""
    searchspace = _make_task_searchspace(
        ["source", "target"], ["target"], TransferLearningMode.POSITIVE_INDEX_KERNEL
    )
    measurements = _make_measurements(["source", "target"], objective)
    surrogate = GaussianProcessSurrogate(kernel_or_factory=MaternKernel())

    with pytest.raises(IncompatibleSurrogateError, match="default kernel"):
        surrogate.fit(searchspace, objective, measurements)


def test_mean_transfer_delegates(objective):
    """A MEAN_TRANSFER override makes the GP delegate to a MeanTransferSurrogate."""
    searchspace = _make_task_searchspace(
        ["source", "target"], ["target"], TransferLearningMode.MEAN_TRANSFER
    )
    measurements = _make_measurements(["source", "target"], objective)
    surrogate = GaussianProcessSurrogate()

    surrogate.fit(searchspace, objective, measurements)

    assert isinstance(surrogate._delegate, MeanTransferSurrogate)
    assert surrogate._delegate._source_gp is not None
    assert surrogate._delegate._target_gp is not None

    candidates = pd.DataFrame({"p": [0.1, 0.5, 0.9], "task": "target"})
    posterior = surrogate.posterior(candidates)
    assert posterior.mean.numel() == 3


def test_mean_transfer_matches_inner_target_gp(objective):
    """The outer posterior equals the inner target GP posterior on stripped inputs."""
    searchspace = _make_task_searchspace(
        ["source", "target"], ["target"], TransferLearningMode.MEAN_TRANSFER
    )
    measurements = _make_measurements(["source", "target"], objective)
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(searchspace, objective, measurements)

    candidates = pd.DataFrame({"p": [0.1, 0.5, 0.9], "task": "target"})
    reduced_candidates = candidates.drop(columns=["task"])

    outer_mean = surrogate.posterior(candidates).mean
    inner_mean = surrogate._delegate._target_gp.posterior(reduced_candidates).mean
    assert torch.allclose(outer_mean, inner_mean, atol=1e-4)


def test_mean_transfer_cache_populated_correctly(objective):
    """Cached training mean equals direct source GP evaluation at training points."""
    searchspace = _make_task_searchspace(
        ["source", "target"], ["target"], TransferLearningMode.MEAN_TRANSFER
    )
    measurements = _make_measurements(["source", "target"], objective)
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(searchspace, objective, measurements)

    delegate = surrogate._delegate
    assert isinstance(delegate, MeanTransferSurrogate)
    target_gp = delegate._target_gp
    assert target_gp is not None
    mean_module = target_gp._model.mean_module

    # Cache must be populated after fitting
    assert mean_module._train_mean_cache is not None

    # Reference: re-evaluate at the same normalized training inputs without the cache
    train_x_norm = target_gp._model.train_inputs[0]
    with torch.no_grad():
        reference = mean_module._eval_source_gp(train_x_norm)

    assert torch.allclose(mean_module._train_mean_cache, reference, atol=1e-6)


@pytest.mark.parametrize(
    ("values", "active_values"),
    [
        pytest.param(["source", "target"], ["source", "target"], id="no_source"),
        pytest.param(["source", "t1", "t2"], ["t1"], id="multiple_sources"),
    ],
)
def test_mean_transfer_requires_single_source_target(values, active_values, objective):
    """Invalid source/target cardinalities raise an error."""
    searchspace = _make_task_searchspace(
        values, active_values, TransferLearningMode.MEAN_TRANSFER
    )
    measurements = _make_measurements(values, objective)
    surrogate = GaussianProcessSurrogate()

    with pytest.raises(
        IncompatibleSearchSpaceError,
        match="one active .target. task value|at most 1 source task",
    ):
        surrogate.fit(searchspace, objective, measurements)


def test_reduced_searchspace_exposes_fit_attributes():
    """The reduced search space exposes the members needed to fit a GP on it."""
    searchspace = _make_task_searchspace(["source", "target"], ["target"])
    reduced = searchspace._drop_parameters({"task"})

    # These must not raise despite the reduced space's restricted attribute access.
    transformed = reduced.transform(pd.DataFrame({"p": [0.5]}), allow_extra=True)
    assert list(transformed.columns) == ["p"]
    assert not reduced.scaling_bounds.empty
    assert reduced.get_comp_rep_parameter_indices("p") == (0,)

    # `task_idx` is whitelisted only because it always returns None here.
    assert reduced.task_idx is None


def test_mean_transfer_campaign_recommend(objective):
    """A default campaign with a MEAN_TRANSFER override can produce recommendations.

    This exercises the full recommender path, where the surrogate is auto-replicated
    into a ``CompositeSurrogate`` and consumed via ``to_botorch``.
    """
    searchspace = _make_task_searchspace(
        ["source", "target"], ["target"], TransferLearningMode.MEAN_TRANSFER
    )
    measurements = _make_measurements(["source", "target"], objective)

    campaign = Campaign(searchspace, objective)
    campaign.add_measurements(measurements)
    recommendation = campaign.recommend(batch_size=3)

    assert len(recommendation) == 3
    # Recommendations are made at the active (target) task only.
    assert set(recommendation["task"]) == {"target"}


@pytest.mark.parametrize(
    ("mode", "expected_propagate"),
    [
        pytest.param(TransferLearningMode.RESIDUAL_LEARNING, False, id="residual"),
        pytest.param(
            TransferLearningMode.RESIDUAL_LEARNING_WITH_UNCERTAINTY,
            True,
            id="residual_unc",
        ),
    ],
)
def test_residual_transfer_delegates(mode, expected_propagate, objective):
    """A residual override makes the GP delegate to a configured ResidualTransfer."""
    searchspace = _make_task_searchspace(["source", "target"], ["target"], mode)
    measurements = _make_measurements(["source", "target"], objective)
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(searchspace, objective, measurements)

    assert isinstance(surrogate._delegate, ResidualTransferSurrogate)
    assert surrogate._delegate.propagate_source_uncertainty is expected_propagate
    assert len(surrogate._delegate._gp_chain) == 2  # source GP + target residual GP


def test_residual_transfer_mean_is_source_plus_residual(objective):
    """The combined posterior mean equals the source mean plus the residual mean."""
    searchspace = _make_task_searchspace(
        ["source", "target"], ["target"], TransferLearningMode.RESIDUAL_LEARNING
    )
    measurements = _make_measurements(["source", "target"], objective)
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(searchspace, objective, measurements)

    candidates = pd.DataFrame({"p": [0.1, 0.5, 0.9], "task": "target"})
    reduced = candidates.drop(columns=["task"])
    delegate = surrogate._delegate

    outer_mean = surrogate.posterior(candidates).mean
    chain_sum = sum(gp.posterior(reduced).mean for gp in delegate._gp_chain)
    assert torch.allclose(outer_mean, chain_sum, atol=1e-4)


def test_residual_transfer_uncertainty_variant_has_larger_variance(objective):
    """Adding source uncertainty yields a variance at least as large everywhere."""
    searchspace = _make_task_searchspace(["source", "target"], ["target"])
    measurements = _make_measurements(["source", "target"], objective)

    residual_only = ResidualTransferSurrogate(propagate_source_uncertainty=False)
    residual_unc = ResidualTransferSurrogate(propagate_source_uncertainty=True)
    residual_only.fit(searchspace, objective, measurements)
    residual_unc.fit(searchspace, objective, measurements)

    candidates = pd.DataFrame({"p": [0.1, 0.5, 0.9], "task": "target"})
    var_only = residual_only.posterior(candidates).variance
    var_unc = residual_unc.posterior(candidates).variance

    assert (var_unc >= var_only - 1e-6).all()
    assert var_unc.sum() > var_only.sum()


def test_residual_transfer_campaign_recommend(objective):
    """A default campaign with a RESIDUAL_LEARNING override can recommend."""
    searchspace = _make_task_searchspace(
        ["source", "target"], ["target"], TransferLearningMode.RESIDUAL_LEARNING
    )
    measurements = _make_measurements(["source", "target"], objective)

    campaign = Campaign(searchspace, objective)
    campaign.add_measurements(measurements)
    recommendation = campaign.recommend(batch_size=3)

    assert len(recommendation) == 3
    assert set(recommendation["task"]) == {"target"}


def test_residual_transfer_rejects_desirability():
    """Residual transfer with a `DesirabilityObjective` raises before fitting."""
    objective = DesirabilityObjective(
        [NumericalTarget("t1"), NumericalTarget("t2")],
        require_normalization=False,
        scalarizer="MEAN",
    )
    searchspace = _make_task_searchspace(["source", "target"], ["target"])
    measurements = _make_measurements(["source", "target"], objective)

    surrogate = ResidualTransferSurrogate()
    with pytest.raises(IncompatibleObjectiveError, match="DesirabilityObjective"):
        surrogate.fit(searchspace, objective, measurements)


def test_residual_transfer_multisource_chain_length(objective):
    """_gp_chain has one entry per source with data plus one for the target."""
    searchspace = _make_task_searchspace(
        ["s1", "s2", "target"], ["target"], TransferLearningMode.RESIDUAL_LEARNING
    )
    measurements = _make_measurements(["s1", "s2", "target"], objective)
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(searchspace, objective, measurements)

    assert len(surrogate._delegate._gp_chain) == 3  # s1 + s2_residual + target_residual


def test_residual_transfer_multisource_mean_is_chain_sum(objective):
    """Posterior mean equals the sum of individual GP means in the chain."""
    searchspace = _make_task_searchspace(
        ["s1", "s2", "target"], ["target"], TransferLearningMode.RESIDUAL_LEARNING
    )
    measurements = _make_measurements(["s1", "s2", "target"], objective)
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(searchspace, objective, measurements)

    candidates = pd.DataFrame({"p": [0.1, 0.5, 0.9], "task": "target"})
    reduced = candidates.drop(columns=["task"])

    outer_mean = surrogate.posterior(candidates).mean
    chain_sum = sum(gp.posterior(reduced).mean for gp in surrogate._delegate._gp_chain)
    assert torch.allclose(outer_mean, chain_sum, atol=1e-4)


def test_residual_transfer_multisource_uncertainty_variant_has_larger_variance(
    objective,
):
    """With propagation, total variance >= residual-only variance (multi-source)."""
    searchspace = _make_task_searchspace(["s1", "s2", "target"], ["target"])
    measurements = _make_measurements(["s1", "s2", "target"], objective)

    residual_only = ResidualTransferSurrogate(propagate_source_uncertainty=False)
    residual_unc = ResidualTransferSurrogate(propagate_source_uncertainty=True)
    residual_only.fit(searchspace, objective, measurements)
    residual_unc.fit(searchspace, objective, measurements)

    candidates = pd.DataFrame({"p": [0.1, 0.5, 0.9], "task": "target"})
    var_only = residual_only.posterior(candidates).variance
    var_unc = residual_unc.posterior(candidates).variance

    assert (var_unc >= var_only - 1e-6).all()
    assert var_unc.sum() > var_only.sum()


def test_reduced_searchspace_rejects_surviving_task():
    """A reduced space that still contains a task parameter raises on construction."""
    searchspace = _make_task_searchspace(["source", "target"], ["target"])
    # Dropping the non-task parameter leaves the task parameter in place.
    with pytest.raises(ValueError, match="must not contain a task parameter"):
        searchspace._drop_parameters({"p"})


def test_mean_transfer_rejects_fantasize_acquisition(objective):
    """A fantasize-requiring acqf (qNIPV) is rejected for a delegating surrogate."""
    searchspace = _make_task_searchspace(
        ["source", "target"], ["target"], TransferLearningMode.MEAN_TRANSFER
    )
    measurements = _make_measurements(["source", "target"], objective)

    recommender = BotorchRecommender(
        acquisition_function=qNegIntegratedPosteriorVariance()
    )
    campaign = Campaign(searchspace, objective, recommender)
    campaign.add_measurements(measurements)

    with pytest.raises(IncompatibleAcquisitionFunctionError, match="fantasiz"):
        campaign.recommend(batch_size=2)


@pytest.mark.parametrize(
    "surrogate_factory",
    [
        pytest.param(MeanTransferSurrogate, id="mean"),
        pytest.param(RGPETransferSurrogate, id="rgpe"),
    ],
)
def test_transfer_surrogate_cold_start_uses_source(surrogate_factory, objective):
    """With no target data, the posterior falls back to the source model."""
    searchspace = _make_task_searchspace(["source", "target"], ["target"])
    # Only source measurements are available (cold start on the target task).
    measurements = _make_measurements(["source"], objective)

    surrogate = surrogate_factory()
    surrogate.fit(searchspace, objective, measurements)

    candidates = pd.DataFrame({"p": [0.1, 0.5, 0.9], "task": "target"})
    reduced = candidates.drop(columns=["task"])
    source_mean = surrogate._source_gp.posterior(reduced).mean

    assert torch.allclose(surrogate.posterior(candidates).mean, source_mean, atol=1e-4)


def test_residual_transfer_cold_start_uses_source_chain(objective):
    """With no target data, residual posterior equals the source chain sum."""
    searchspace = _make_task_searchspace(["source", "target"], ["target"])
    measurements = _make_measurements(["source"], objective)

    surrogate = ResidualTransferSurrogate()
    surrogate.fit(searchspace, objective, measurements)

    candidates = pd.DataFrame({"p": [0.1, 0.5, 0.9], "task": "target"})
    reduced = candidates.drop(columns=["task"])
    # Cold start: only one source GP in the chain.
    assert len(surrogate._gp_chain) == 1
    chain_mean = surrogate._gp_chain[0].posterior(reduced).mean
    assert torch.allclose(surrogate.posterior(candidates).mean, chain_mean, atol=1e-4)


def test_rgpe_delegates(objective):
    """An RGPE override makes the GP delegate to an RGPETransferSurrogate."""
    searchspace = _make_task_searchspace(
        ["s1", "s2", "target"], ["target"], TransferLearningMode.RGPE
    )
    measurements = _make_measurements(["s1", "s2", "target"], objective)
    surrogate = GaussianProcessSurrogate()

    surrogate.fit(searchspace, objective, measurements)

    delegate = surrogate._delegate
    assert isinstance(delegate, RGPETransferSurrogate)
    assert len(delegate._source_gps) == 2
    assert delegate._target_gp is not None

    candidates = pd.DataFrame({"p": [0.1, 0.5, 0.9], "task": "target"})
    posterior = surrogate.posterior(candidates)
    assert posterior.mean.numel() == 3
    assert (posterior.variance > 0).all()


def test_rgpe_weights_are_a_convex_combination(objective):
    """The RGPE weights are non-negative and sum to one over all ensemble members."""
    searchspace = _make_task_searchspace(
        ["s1", "s2", "target"], ["target"], TransferLearningMode.RGPE
    )
    measurements = _make_measurements(["s1", "s2", "target"], objective)
    surrogate = RGPETransferSurrogate()
    surrogate.fit(searchspace, objective, measurements)

    weights = surrogate._weights
    # One weight per source plus one for the target model.
    assert weights.numel() == 3
    assert (weights >= 0).all()
    assert abs(weights.sum().item() - 1.0) < 1e-6


def test_rgpe_cold_start_uses_uniform_source_weights(objective):
    """With no target data, RGPE averages the source models uniformly."""
    searchspace = _make_task_searchspace(["s1", "s2", "target"], ["target"])
    measurements = _make_measurements(["s1", "s2"], objective)
    surrogate = RGPETransferSurrogate()
    surrogate.fit(searchspace, objective, measurements)

    assert surrogate._target_gp is None
    assert torch.allclose(surrogate._weights, torch.full((2,), 0.5))


def test_rgpe_single_target_point_uses_uniform_weights(objective):
    """With a single target point, RGPE averages all models uniformly."""
    searchspace = _make_task_searchspace(["s1", "s2", "target"], ["target"])
    measurements = _make_measurements(["s1", "s2"], objective)
    target_point = pd.DataFrame({"p": [0.4], "task": ["target"], "t": [0.5]})
    measurements = pd.concat([measurements, target_point], ignore_index=True)

    surrogate = RGPETransferSurrogate()
    surrogate.fit(searchspace, objective, measurements)

    assert surrogate._target_gp is not None
    assert torch.allclose(surrogate._weights, torch.full((3,), 1.0 / 3.0))


def test_rgpe_campaign_recommend(objective):
    """A default campaign with an RGPE override can produce recommendations."""
    searchspace = _make_task_searchspace(
        ["s1", "s2", "target"], ["target"], TransferLearningMode.RGPE
    )
    measurements = _make_measurements(["s1", "s2", "target"], objective)

    campaign = Campaign(searchspace, objective)
    campaign.add_measurements(measurements)
    recommendation = campaign.recommend(batch_size=3)

    assert len(recommendation) == 3
    assert set(recommendation["task"]) == {"target"}
