"""Tests for the rank-weighted Gaussian process ensemble (RGPE)."""

from copy import deepcopy
from functools import partial

import pandas as pd
import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from pytest import param

from baybe.acquisition import EI, qKG, qLogEI, qLogNEHVI, qLogNEI, qNIPV
from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.objectives import ParetoObjective
from baybe.parameters import NumericalContinuousParameter, TaskParameter
from baybe.recommenders import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process._rgpe import RGPEModel, make_rgpe_model
from baybe.surrogates.gaussian_process.core import _ModelContext
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import to_tensor
from tests.conftest import FAST_OPTIMIZATION_SETTINGS


def _fit_gp(
    x: torch.Tensor, y: torch.Tensor, input_transform: Normalize | None = None
) -> SingleTaskGP:
    """Fit a GP with normalized inputs and standardized outputs."""
    if input_transform is None:
        bounds = torch.tensor([[0.0], [5.0]], dtype=x.dtype)
        input_transform = Normalize(d=1, bounds=bounds)
    gp = SingleTaskGP(
        x,
        y,
        input_transform=deepcopy(input_transform),
        outcome_transform=Standardize(1),
        covar_module=ScaleKernel(MaternKernel()),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp


@pytest.fixture(name="members", scope="module")
def fixture_members() -> tuple[SingleTaskGP, ...]:
    """Two sources and a target (last) on clearly different output scales."""
    torch.manual_seed(0)
    x = torch.linspace(0, 5, 8, dtype=torch.float64).unsqueeze(-1)
    ys = (100 * torch.sin(x) + 50, -10 * torch.cos(x), torch.sin(x) + 1)
    return tuple(_fit_gp(x, y + 0.1 * torch.randn_like(y)) for y in ys)


def _get_noise(model: SingleTaskGP, X: torch.Tensor) -> torch.Tensor:
    """Get the observation noise a model adds to its own posterior."""
    noisy = model.posterior(X, observation_noise=True).variance
    return (noisy - model.posterior(X).variance).squeeze(-1)


@pytest.mark.parametrize(
    "shape", [(4, 1), (3, 2, 1), (2, 3, 2, 1)], ids=["n", "b_q", "s_b_q"]
)
def test_posterior_combines_members(members, shape):
    """The posterior is the weighted sum of the members' posteriors."""
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    model = RGPEModel(members, weights, target_index=2)
    X = 5 * torch.rand(shape, dtype=torch.float64)

    posterior = model.posterior(X)
    posteriors = [m.posterior(X) for m in members]
    mean = sum(w * p.mean for w, p in zip(weights, posteriors))
    cov = sum(
        w**2 * p.distribution.covariance_matrix for w, p in zip(weights, posteriors)
    )
    assert torch.allclose(posterior.mean, mean)
    assert torch.allclose(posterior.distribution.covariance_matrix, cov)


@pytest.mark.parametrize("noise", ["target", "tensor", "cold_start"])
def test_observation_noise(members, noise):
    """Observation noise is added once, in original outcome units."""
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    X = torch.tensor([[1.0], [2.5]], dtype=torch.float64)
    match noise:
        case "target":
            model = RGPEModel(members, weights, target_index=2)
            observation_noise, expected = True, _get_noise(members[2], X)
        case "tensor":
            model = RGPEModel(members, weights, target_index=2)
            observation_noise = torch.tensor(0.3, dtype=torch.float64)
            expected = observation_noise.expand(2)
        case "cold_start":
            model = RGPEModel(members, weights)
            observation_noise = True
            expected = sum(w * _get_noise(m, X) for w, m in zip(weights, members))

    added = model.posterior(X, observation_noise=observation_noise).variance
    added = (added - model.posterior(X).variance).squeeze(-1)
    assert torch.allclose(added, expected)


def test_pruning_keeps_target(members):
    """Zero-weight sources are pruned, a zero-weight target is kept."""
    weights = torch.tensor([0.6, 0.0, 0.0], dtype=torch.float64)
    model = RGPEModel.from_ranked_models(members, weights, target_index=2)
    X = torch.tensor([[1.0]], dtype=torch.float64)

    assert model._members == (members[0], members[2])
    assert model.target_index == 1
    assert torch.equal(model.weights, torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert torch.allclose(model.posterior(X).mean, members[0].posterior(X).mean)
    noisy = model.posterior(X, observation_noise=True).variance
    noise = (noisy - model.posterior(X).variance).squeeze(-1)
    assert torch.allclose(noise, _get_noise(members[2], X))


def test_posterior_arguments(members):
    """Positional arguments follow BoTorch's order and only output 0 exists."""
    model = RGPEModel(members, torch.full((3,), 1 / 3, dtype=torch.float64))
    X = torch.tensor([[1.0]], dtype=torch.float64)

    positional = model.posterior(X, [0], True).variance
    assert torch.equal(positional, model.posterior(X, observation_noise=True).variance)
    with pytest.raises(ValueError, match="single output"):
        model.posterior(X, output_indices=[1])


@pytest.mark.parametrize("wrap", [False, True], ids=["plain", "model_list"])
def test_fantasize(members, wrap):
    """Fantasizing conditions every member on normalized fantasy inputs."""
    model = RGPEModel(members, torch.full((3,), 1 / 3, dtype=torch.float64))
    X = torch.tensor([[4.0]], dtype=torch.float64)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([4]))

    fantasy = (ModelListGP(model) if wrap else model).fantasize(X, sampler)
    rgpe = fantasy.models[0] if wrap else fantasy
    assert fantasy.posterior(X).mean.shape == (4, 1, 1)
    assert (fantasy.posterior(X).variance < model.posterior(X).variance).all()
    for m in rgpe._members:
        assert torch.allclose(
            m.train_inputs[0][..., -1, :], torch.tensor([0.8]).double()
        )


def test_model_contract(members):
    """The ensemble is a GPyTorch model without a single training dataset."""
    model = RGPEModel(members, torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64))
    restored = deepcopy(model)
    restored.weights.fill_(1 / 3)
    restored.load_state_dict(model.state_dict())
    X = torch.tensor([[1.0], [3.0]], dtype=torch.float64)

    assert isinstance(model, GPyTorchModel)
    assert not hasattr(model, "train_inputs")
    assert torch.equal(restored.posterior(X).mean, model.posterior(X).mean)


def _make_data(sources: dict[str, int], n_target: int) -> tuple:
    """Make a model context, its training tensors and a matching GP fitter."""
    values = ("T", *sources)
    searchspace = SearchSpace.from_product(
        [
            NumericalContinuousParameter("x", (0, 5)),
            TaskParameter("task", values, active_values=("T",)),
        ]
    )
    counts = {"T": n_target, **sources}
    measurements = pd.DataFrame(
        {
            "x": [0.5 + 0.7 * i for c in counts.values() for i in range(c)],
            "task": [t for t, c in counts.items() for _ in range(c)],
        }
    )
    measurements["y"] = measurements["x"] ** 2
    objective = NumericalTarget("y").to_objective()
    context = _ModelContext(searchspace, objective, measurements)
    train_x = to_tensor(searchspace.transform(measurements, allow_extra=True))
    train_y = to_tensor(measurements[["y"]])
    input_transform = GaussianProcessSurrogate._make_input_transform(context)
    fit_gp = partial(_fit_gp, input_transform=input_transform)
    return context, train_x, train_y, fit_gp


@pytest.mark.parametrize(
    ("sources", "n_target", "n_members", "is_uniform"),
    [
        param({"A": 5}, 0, 1, True, id="cold_start"),
        param({"A": 5}, 1, 2, True, id="one_target"),
        param({"A": 5, "B": 0, "C": 4}, 4, None, False, id="ranked"),
    ],
)
def test_make_rgpe_model(sources, n_target, n_members, is_uniform):
    """Members are fitted per task with data and weighted as specified."""
    context, train_x, train_y, fit_gp = _make_data(sources, n_target)
    model = make_rgpe_model(context, train_x, train_y, fit_gp)

    assert model.weights.dtype == train_x.dtype
    assert torch.isclose(model.weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    if is_uniform:
        assert len(model._members) == n_members
        assert torch.allclose(model.weights, model.weights[0])
    assert model.target_index == (None if n_target == 0 else len(model._members) - 1)


def test_make_rgpe_model_requires_source_data():
    """Fitting without source data raises."""
    context, train_x, train_y, fit_gp = _make_data({"A": 0}, 3)
    with pytest.raises(IncompatibleSearchSpaceError, match="at least one source"):
        make_rgpe_model(context, train_x, train_y, fit_gp)


_SURROGATE_MEASUREMENTS = pd.DataFrame(
    {
        "x": [0.5, 1.5, 2.5, 3.5, 4.5, 0.3, 1.2, 2.2, 3.3, 1.0, 2.0, 3.0, 4.0],
        "y": [0.8, 2.3, 2.7, 4.1, 3.6, 4.0, 3.1, 2.2, 1.4, 1.1, 2.4, 2.9, 3.5],
        "z": [1.0, 0.5, 0.2, 0.4, 0.9, 0.1, 0.3, 0.8, 0.6, 0.7, 0.2, 0.5, 0.4],
        "task": ["A"] * 5 + ["B"] * 4 + ["T"] * 4,
    }
)
"""Measurements of two source tasks and a target task."""

_CANDIDATES = pd.DataFrame({"x": [0.25, 1.25, 2.6, 3.75, 4.9], "task": "T"})
_OBJECTIVE = NumericalTarget("y").to_objective()


def _make_searchspace(mode: str | None = "RGPE") -> SearchSpace:
    """Make a search space with a continuous parameter and a task parameter."""
    task = TaskParameter(
        "task",
        ("T", "A", "B"),
        active_values=("T",),
        override_transfer_learning_mode=mode,
    )
    return SearchSpace.from_product([NumericalContinuousParameter("x", (0, 5)), task])


def _fit(measurements: pd.DataFrame, mode: str | None = "RGPE", **kwargs):
    """Fit a GP surrogate on the given measurements."""
    gp = GaussianProcessSurrogate(**kwargs)
    gp.fit(_make_searchspace(mode), _OBJECTIVE, measurements)
    return gp


@pytest.mark.parametrize("n_target", [0, 1], ids=["cold_start", "one_target"])
def test_members_match_naive_gps(n_target):
    """Each member equals a naive GP fitted on its task's data alone."""
    measurements = _SURROGATE_MEASUREMENTS.groupby("task").head(5 if n_target else 4)
    measurements = measurements[measurements["task"] != "T"]
    if n_target:
        target = _SURROGATE_MEASUREMENTS[_SURROGATE_MEASUREMENTS["task"] == "T"]
        measurements = pd.concat([measurements, target.head(n_target)])
    rgpe = _fit(measurements).to_botorch()

    tasks = ["A", "B", "T"][: len(rgpe._members)]
    for member, task in zip(rgpe._members, tasks, strict=True):
        naive = _fit(measurements[measurements["task"] == task], "NAIVE").to_botorch()
        expected = naive.state_dict()
        assert member.state_dict().keys() == expected.keys()
        for key, value in member.state_dict().items():
            assert torch.equal(value, expected[key])


def test_refit_replaces_model():
    """Refitting without RGPE replaces the ensemble."""
    gp = _fit(_SURROGATE_MEASUREMENTS)
    assert isinstance(gp.to_botorch(), RGPEModel)
    gp.fit(_make_searchspace("NAIVE"), _OBJECTIVE, _SURROGATE_MEASUREMENTS)
    assert isinstance(gp.to_botorch(), SingleTaskGP)


def test_members_share_no_modules():
    """Members are built from copies of the resolved components."""
    first, second, *_ = _fit(_SURROGATE_MEASUREMENTS).to_botorch()._members
    for name in ("input_transform", "mean_module", "covar_module", "likelihood"):
        assert getattr(first, name) is not getattr(second, name)


def test_default_path_uses_raw_modules():
    """Without RGPE, a raw GPyTorch module is used as is."""
    kernel = MaternKernel()
    gp = GaussianProcessSurrogate(kernel_or_factory=kernel)
    searchspace = SearchSpace.from_product([NumericalContinuousParameter("x", (0, 5))])
    gp.fit(searchspace, _OBJECTIVE, _SURROGATE_MEASUREMENTS.drop(columns="task"))
    assert gp.to_botorch().covar_module is kernel


def test_posterior_mean_function():
    """The posterior mean function reproduces the ensemble's posterior mean."""
    gp = _fit(_SURROGATE_MEASUREMENTS)
    searchspace = _make_searchspace()
    mean_module = gp.posterior_mean_function(
        searchspace, _OBJECTIVE, _SURROGATE_MEASUREMENTS
    )
    context = _ModelContext(searchspace, _OBJECTIVE, _SURROGATE_MEASUREMENTS)
    input_transform = GaussianProcessSurrogate._make_input_transform(context)
    outcome_transform = GaussianProcessSurrogate._make_outcome_transform(
        to_tensor(_SURROGATE_MEASUREMENTS[["y"]])
    )
    outcome_transform.eval()

    X = to_tensor(searchspace.transform(_CANDIDATES))
    standardized = mean_module(input_transform(X)).unsqueeze(-1)
    mean = outcome_transform.untransform(standardized)[0]
    assert torch.allclose(mean, gp.to_botorch().posterior(X).mean)


def test_deepcopy_keeps_posterior():
    """A deep copy of a fitted surrogate predicts the same."""
    gp = _fit(_SURROGATE_MEASUREMENTS)
    expected = gp.posterior(_CANDIDATES).mean
    assert torch.equal(deepcopy(gp).posterior(_CANDIDATES).mean, expected)


def test_several_active_values_raise():
    """RGPE requires exactly one active value."""
    with pytest.raises(ValueError, match="exactly one value in 'active_values'"):
        TaskParameter(
            "task",
            ("T", "A"),
            active_values=("T", "A"),
            override_transfer_learning_mode="RGPE",
        )


def test_missing_source_data_raises():
    """Fitting without source data raises."""
    target = _SURROGATE_MEASUREMENTS[_SURROGATE_MEASUREMENTS["task"] == "T"]
    with pytest.raises(IncompatibleSearchSpaceError, match="at least one source"):
        _fit(target)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("acqf", "batch_size", "targets"),
    [
        param(EI(), 1, ["y"], id="EI"),
        param(qLogEI(), 2, ["y"], id="qLogEI"),
        param(qLogNEI(), 2, ["y"], id="qLogNEI"),
        param(qKG(), 1, ["y"], id="qKG"),
        param(qNIPV(), 2, ["y"], id="qNIPV"),
        param(qLogNEHVI(), 2, ["y", "z"], id="qLogNEHVI"),
    ],
)
def test_recommendation(acqf, batch_size, targets):
    """A campaign in RGPE mode recommends target-task points."""
    objective = (
        ParetoObjective([NumericalTarget(t) for t in targets])
        if len(targets) > 1
        else _OBJECTIVE
    )
    recommender = BotorchRecommender(
        acquisition_function=acqf, **FAST_OPTIMIZATION_SETTINGS
    )
    recommendations = recommender.recommend(
        batch_size, _make_searchspace(), objective, _SURROGATE_MEASUREMENTS
    )
    assert (recommendations["task"] == "T").all()
