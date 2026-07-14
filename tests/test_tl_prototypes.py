"""Tests for the transfer-learning prototype dispatching (Case A + MEAN_TRANSFER)."""

import numpy as np
import pandas as pd
import pytest
import torch

from baybe.campaign import Campaign
from baybe.exceptions import IncompatibleSearchSpaceError, IncompatibleSurrogateError
from baybe.kernels.basic import IndexKernel, MaternKernel, PositiveIndexKernel
from baybe.parameters.categorical import TaskParameter
from baybe.parameters.enum import TransferLearningMode
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.presets.baybe import _BayBETaskKernelFactory
from baybe.surrogates.transfer_learning.mean_transfer import MeanTransferSurrogate
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

    with pytest.raises(IncompatibleSearchSpaceError, match="one source and one target"):
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
