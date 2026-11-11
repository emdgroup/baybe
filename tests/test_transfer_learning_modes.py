"""Tests for the naive and the rank-weighted ensemble (RGPE) transfer learning modes."""

from copy import deepcopy
from functools import partial
from unittest.mock import patch

import botorch.fit
import pandas as pd
import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelList, SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from pytest import param
from torch.testing import assert_close

from baybe.exceptions import IncompatibleOverrideError, NoMeasurementsError
from baybe.parameters import NumericalContinuousParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process._rgpe import RGPEModel
from baybe.surrogates.gaussian_process.core import _ModelContext
from baybe.surrogates.gaussian_process.presets.edbo import EDBOKernelFactory
from baybe.surrogates.gaussian_process.presets.hvarfner import (
    HvarfnerLikelihoodFactory,
    HvarfnerMeanFactory,
)
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import to_tensor

_TASKS = ("T", "A", "B")
"""The task values, with the target task first."""

_OBJECTIVE = NumericalTarget("y").to_objective()


def _make_gp(
    x: torch.Tensor, y: torch.Tensor, input_transform: Normalize | None = None
) -> SingleTaskGP:
    """Make an unfitted GP with normalized inputs and standardized outputs."""
    if input_transform is None:
        bounds = torch.tensor([[0.0], [5.0]], dtype=x.dtype)
        input_transform = Normalize(d=1, bounds=bounds)
    return SingleTaskGP(
        x,
        y,
        input_transform=deepcopy(input_transform),
        outcome_transform=Standardize(1),
        covar_module=ScaleKernel(MaternKernel()),
    )


def _make_searchspace(mode: str = "RGPE") -> SearchSpace:
    """Make a search space with a continuous parameter and a task parameter."""
    task = TaskParameter(
        "task", _TASKS, active_values=("T",), override_transfer_learning_mode=mode
    )
    return SearchSpace.from_product([NumericalContinuousParameter("x", (0, 5)), task])


def _make_measurements(**n_rows: int) -> pd.DataFrame:
    """Make measurements with the given number of rows per task.

    The tasks share the inputs but are offset in their outputs.
    """
    x = [0.5 + 0.7 * i for n in n_rows.values() for i in range(n)]
    tasks = [t for t, n in n_rows.items() for _ in range(n)]
    offsets = [10 * _TASKS.index(t) for t in tasks]
    return pd.DataFrame(
        {"x": x, "task": tasks, "y": [xi**2 + o for xi, o in zip(x, offsets)]}
    )


_MEASUREMENTS = _make_measurements(A=3, B=3, T=3)
"""Measurements of two source tasks and the target task."""


def _fit(measurements: pd.DataFrame, mode: str = "RGPE", **kwargs):
    """Fit a GP surrogate on the given measurements."""
    gp = GaussianProcessSurrogate(**kwargs)
    gp.fit(_make_searchspace(mode), _OBJECTIVE, measurements)
    return gp


@pytest.fixture(name="no_fit")
def fixture_no_fit(request):
    """Skip the hyperparameter optimization of the surrogate's GPs."""
    with patch.object(
        botorch.fit, "fit_gpytorch_mll", side_effect=lambda mll, **_: mll
    ) as patched:
        yield patched
    # Ensures the patch is not silently bypassed. Expected failures may raise early.
    if request.node.get_closest_marker("xfail") is None:
        patched.assert_called()


_REDUCED_SEARCHSPACE_UNSUPPORTED = pytest.mark.xfail(
    raises=IncompatibleOverrideError,
    reason="Factories requiring the comp rep of the reduced search space",
)

_SURROGATE_FACTORIES = [
    # The default kernel is reduced by a dedicated code path
    param(GaussianProcessSurrogate, id="default"),
    # Other kernel factories are called on the reduced search space. Hvarfner
    # components change with the number of tasks, revealing whether they also
    # received the reduced search space.
    param(
        partial(
            GaussianProcessSurrogate,
            kernel_or_factory=EDBOKernelFactory(),
            mean_or_factory=HvarfnerMeanFactory(),
            likelihood_or_factory=HvarfnerLikelihoodFactory(),
        ),
        id="factories",
    ),
    *(
        param(partial(GaussianProcessSurrogate.from_preset, p), id=p)
        for p in ("CHEN", "EDBO", "EDBO_SMOOTHED")
    ),
    *(
        param(
            partial(GaussianProcessSurrogate.from_preset, p),
            id=p,
            marks=_REDUCED_SEARCHSPACE_UNSUPPORTED,
        )
        for p in ("HVARFNER", "BOTORCH")
    ),
]
"""Surrogate factories whose components must be made for the reduced search space."""


@pytest.mark.usefixtures("no_fit")
@pytest.mark.parametrize("make_surrogate", _SURROGATE_FACTORIES)
def test_naive_mode_matches_task_free_gp(make_surrogate):
    """The naive mode yields the same GP as dropping the task parameter."""
    searchspace_naive = _make_searchspace("NAIVE")
    searchspace_free = NumericalContinuousParameter("x", (0, 5)).to_searchspace()
    naive = make_surrogate()
    naive.fit(searchspace_naive, _OBJECTIVE, _MEASUREMENTS)
    task_free = make_surrogate()
    task_free.fit(searchspace_free, _OBJECTIVE, _MEASUREMENTS.drop(columns="task"))

    # Same model state, where kernel dimensions are compared via their column names
    # since the task column shifts the index of the continuous parameter
    state_naive = naive.to_botorch().state_dict()
    state_free = task_free.to_botorch().state_dict()
    columns_naive = searchspace_naive.comp_rep_columns
    columns_free = searchspace_free.comp_rep_columns
    assert state_naive.keys() == state_free.keys()
    for key, value in state_free.items():
        if key.startswith("input_transform"):
            continue
        if key.endswith("active_dims"):
            assert [columns_naive[i] for i in state_naive[key]] == [
                columns_free[i] for i in value
            ]
        else:
            assert_close(state_naive[key], value)

    # Same predictions, irrespective of the task label
    candidates = pd.DataFrame({"x": [0.25, 2.6, 4.9]})
    posterior_free = task_free.posterior(candidates)
    for task in _TASKS:
        posterior_naive = naive.posterior(candidates.assign(task=task))
        assert_close(posterior_naive.mean, posterior_free.mean)
        assert_close(
            posterior_naive.distribution.covariance_matrix,
            posterior_free.distribution.covariance_matrix,
        )


@pytest.fixture(name="members", scope="module")
def fixture_members() -> tuple[SingleTaskGP, ...]:
    """Two sources and a target (last) on clearly different output scales."""
    torch.manual_seed(0)
    x = torch.linspace(0, 5, 8, dtype=torch.float64).unsqueeze(-1)
    ys = (100 * torch.sin(x) + 50, -10 * torch.cos(x), torch.sin(x) + 1)
    members = tuple(_make_gp(x, y + 0.1 * torch.randn_like(y)) for y in ys)
    for gp in members:
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return members


def _get_noise(model: SingleTaskGP, X: torch.Tensor) -> torch.Tensor:
    """Get the observation noise a model adds to its own posterior."""
    noisy = model.posterior(X, observation_noise=True).variance
    return (noisy - model.posterior(X).variance).squeeze(-1)


def _make_model(members, weights) -> RGPEModel:
    """Make an ensemble whose target is the last member."""
    return RGPEModel(members, weights, target_index=len(members) - 1)


def _rescale(model: SingleTaskGP, scale: float, shift: float) -> SingleTaskGP:
    """Get a copy of a GP as if fitted on affinely transformed outputs."""
    model = deepcopy(model)
    transform = model.outcome_transform
    transform.means = scale * transform.means + shift
    transform.stdvs = scale * transform.stdvs
    transform._stdvs_sq = transform.stdvs**2
    return model


@pytest.mark.parametrize("shape", [(4, 1), (2, 3, 2, 1)], ids=["n", "s_b_q"])
def test_posterior_combines_standardized_members(members, shape):
    """The posterior combines standardized members in target units."""
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    model = _make_model(members, weights)
    X = 5 * torch.rand(shape, dtype=torch.float64)

    posterior = model.posterior(X)
    target_tf = members[-1].outcome_transform
    mean, cov = 0, 0
    for w, m in zip(weights, members):
        p, tf = m.posterior(X), m.outcome_transform
        mean = mean + w * (p.mean - tf.means) / tf.stdvs
        cov = cov + w**2 * p.distribution.covariance_matrix / tf.stdvs.squeeze() ** 2
    assert torch.allclose(posterior.mean, target_tf.means + target_tf.stdvs * mean)
    assert torch.allclose(
        posterior.distribution.covariance_matrix, target_tf.stdvs.squeeze() ** 2 * cov
    )


@pytest.mark.parametrize("observation_noise", [False, True], ids=["noiseless", "noisy"])
def test_target_only_matches_target_gp(members, observation_noise):
    """With all weight on the target, the ensemble behaves like the target GP."""
    model = _make_model(members, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    target = members[-1]

    X = torch.tensor([[0.3], [2.2], [4.1]], dtype=torch.float64)
    actual = model.posterior(X, observation_noise=observation_noise)
    expected = target.posterior(X, observation_noise=observation_noise)
    assert torch.allclose(actual.mean, expected.mean)
    assert torch.allclose(actual.variance, expected.variance)


@pytest.mark.parametrize("member", [0, 2], ids=["source", "target"])
def test_affine_output_maps(members, member):
    """Source output maps leave the posterior unchanged, target maps carry over."""
    scale, shift = 0.5, -7.0
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    mapped = list(members)
    mapped[member] = _rescale(members[member], scale, shift)
    model, mapped_model = _make_model(members, weights), _make_model(mapped, weights)
    X = torch.tensor([[0.3], [2.2], [4.1]], dtype=torch.float64)

    a, c = (scale, shift) if member == 2 else (1.0, 0.0)
    posterior = model.posterior(X, observation_noise=True)
    mapped_posterior = mapped_model.posterior(X, observation_noise=True)
    assert torch.allclose(mapped_posterior.mean, a * posterior.mean + c)
    assert torch.allclose(
        mapped_posterior.distribution.covariance_matrix,
        a**2 * posterior.distribution.covariance_matrix,
    )


@pytest.mark.parametrize("noise", ["target", "tensor"])
def test_observation_noise(members, noise):
    """Observation noise is added once, in the transformed target space."""
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    X = torch.tensor([[1.0], [2.5]], dtype=torch.float64)
    model = _make_model(members, weights)
    match noise:
        case "target":
            observation_noise, expected = True, _get_noise(members[2], X)
        case "tensor":
            observation_noise = torch.tensor(0.3, dtype=torch.float64)
            stdv_sq = members[2].outcome_transform.stdvs.squeeze() ** 2
            expected = (observation_noise * stdv_sq).expand(2)

    added = model.posterior(X, observation_noise=observation_noise).variance
    added = (added - model.posterior(X).variance).squeeze(-1)
    assert torch.allclose(added, expected)


def test_pruning_keeps_target(members):
    """Zero-weight sources are pruned, a zero-weight target is kept."""
    weights = torch.tensor([0.6, 0.0, 0.0], dtype=torch.float64)
    model = RGPEModel.from_ranked_models(members, weights, target_index=2)

    assert model._members == (members[0], members[2])
    assert model.target_index == 1
    assert torch.equal(model.weights, torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_posterior_arguments(members):
    """Positional arguments follow BoTorch's order and only output 0 exists."""
    model = _make_model(members, torch.full((3,), 1 / 3, dtype=torch.float64))
    X = torch.tensor([[1.0]], dtype=torch.float64)

    positional = model.posterior(X, [0], True).variance
    assert torch.equal(positional, model.posterior(X, observation_noise=True).variance)
    with pytest.raises(ValueError, match="single output"):
        model.posterior(X, output_indices=[1])


def test_model_list_posterior(members):
    """Ensembles combine into a model list, as used for multiple targets."""
    model = _make_model(members, torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64))
    other = _make_model(members, torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64))
    X = torch.tensor([[1.0], [3.0]], dtype=torch.float64)

    posterior = ModelList(model, other).posterior(X)
    expected = torch.cat([model.posterior(X).mean, other.posterior(X).mean], dim=-1)
    assert torch.allclose(posterior.mean, expected)


@pytest.mark.parametrize(
    ("condition", "match"),
    [
        param(
            lambda m, X: m.condition_on_observations(X, X),
            "not implemented",
            id="condition",
        ),
        param(
            lambda m, X: m.fantasize(X, None),
            "does not support fantasizing",
            id="fantasize",
        ),
        param(
            lambda m, X: ModelList(m).fantasize(X, None),
            "does not support fantasizing",
            id="model_list",
        ),
    ],
)
def test_conditioning_raises(members, condition, match):
    """Conditioning on observations and fantasizing are not supported."""
    model = _make_model(members, torch.full((3,), 1 / 3, dtype=torch.float64))
    X = torch.tensor([[4.0]], dtype=torch.float64)
    with pytest.raises(NotImplementedError, match=match):
        condition(model, X)


def test_model_contract(members):
    """The ensemble is a BoTorch model without a single training dataset."""
    model = _make_model(members, torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64))
    restored = deepcopy(model)
    restored.weights.fill_(1 / 3)
    restored.load_state_dict(model.state_dict())
    X = torch.tensor([[1.0], [3.0]], dtype=torch.float64)

    assert isinstance(model, Model)
    assert not hasattr(model, "train_inputs")
    assert torch.equal(restored.posterior(X).mean, model.posterior(X).mean)


@pytest.mark.parametrize(
    "n_rows",
    [
        param({"A": 3, "B": 3}, id="cold_start"),
        param({"A": 3, "T": 1}, id="one_target"),
        param({"A": 3, "B": 3, "T": 3}, id="ranked"),
    ],
)
def test_from_context(n_rows):
    """One member is made per task with data, the target last, and weighted."""
    measurements = _make_measurements(**n_rows)
    searchspace = _make_searchspace()
    context = _ModelContext(searchspace, _OBJECTIVE, measurements)
    train_x = to_tensor(searchspace.transform(measurements, allow_extra=True))
    train_y = to_tensor(measurements[["y"]])
    input_transform = GaussianProcessSurrogate._make_input_transform(context)
    received = []

    def make_gp(x: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
        received.append(y)
        return _make_gp(x, y, input_transform)

    model = RGPEModel.from_context(context, train_x, train_y, make_gp)

    # Sources in task order, then the target
    tasks = [t for t in (*_TASKS[1:], "T") if t in n_rows]
    for y, task in zip(received, tasks, strict=True):
        expected = to_tensor(measurements.loc[measurements["task"] == task, ["y"]])
        assert torch.equal(y, expected)

    assert model.weights.dtype == train_x.dtype
    assert torch.isclose(model.weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    if n_rows.get("T", 0) < 2:
        assert len(model._members) == len(tasks)
        assert torch.allclose(model.weights, model.weights[0])
    if "T" in n_rows:
        assert model.target_index == len(model._members) - 1
        assert model.outcome_transform is model._members[-1].outcome_transform
    else:
        assert model.target_index is None


def test_missing_source_data_raises():
    """Fitting without source data raises."""
    with pytest.raises(NoMeasurementsError, match="at least one source"):
        _fit(_make_measurements(T=3))


@pytest.mark.usefixtures("no_fit")
@pytest.mark.parametrize("make_surrogate", _SURROGATE_FACTORIES)
def test_members_match_naive_gps(make_surrogate):
    """Each member equals a naive GP made from its task's data alone."""
    rgpe = make_surrogate()
    rgpe.fit(_make_searchspace("RGPE"), _OBJECTIVE, _MEASUREMENTS)

    for member, task in zip(rgpe.to_botorch()._members, ("A", "B", "T"), strict=True):
        naive = make_surrogate()
        task_data = _MEASUREMENTS[_MEASUREMENTS["task"] == task]
        naive.fit(_make_searchspace("NAIVE"), _OBJECTIVE, task_data)
        expected = naive.to_botorch().state_dict()
        assert member.state_dict().keys() == expected.keys()
        for key, value in member.state_dict().items():
            assert torch.equal(value, expected[key])


@pytest.mark.usefixtures("no_fit")
def test_members_get_copied_components():
    """Each member gets its own copy of the components."""
    mean, likelihood = ConstantMean(), GaussianLikelihood()
    model = _fit(
        _MEASUREMENTS, mean_or_factory=mean, likelihood_or_factory=likelihood
    ).to_botorch()

    for name in ("input_transform", "mean_module", "covar_module", "likelihood"):
        modules = [getattr(m, name) for m in model._members]
        assert len({id(m) for m in modules}) == len(modules)
    assert all(m.mean_module is not mean for m in model._members)
    assert all(m.likelihood is not likelihood for m in model._members)


@pytest.mark.usefixtures("no_fit")
def test_refit_replaces_model():
    """Refitting without RGPE replaces the ensemble."""
    gp = _fit(_MEASUREMENTS)
    assert isinstance(gp.to_botorch(), RGPEModel)
    gp.fit(_make_searchspace("NAIVE"), _OBJECTIVE, _MEASUREMENTS)
    assert isinstance(gp.to_botorch(), SingleTaskGP)


@pytest.mark.usefixtures("no_fit")
def test_posterior_mean_function():
    """The posterior mean function reproduces the ensemble's posterior mean."""
    gp = _fit(_MEASUREMENTS)
    searchspace = _make_searchspace()
    mean_module = gp.posterior_mean_function(searchspace, _OBJECTIVE, _MEASUREMENTS)
    context = _ModelContext(searchspace, _OBJECTIVE, _MEASUREMENTS)
    input_transform = GaussianProcessSurrogate._make_input_transform(context)
    outcome_transform = GaussianProcessSurrogate._make_outcome_transform(
        to_tensor(_MEASUREMENTS[["y"]])
    )
    outcome_transform.eval()

    candidates = pd.DataFrame({"x": [0.25, 2.6, 4.9], "task": "T"})
    X = to_tensor(searchspace.transform(candidates))
    standardized = mean_module(input_transform(X)).unsqueeze(-1)
    mean = outcome_transform.untransform(standardized)[0]
    assert torch.allclose(mean, gp.to_botorch().posterior(X).mean)


@pytest.mark.usefixtures("no_fit")
def test_deepcopy_keeps_posterior():
    """A deep copy of a fitted surrogate predicts the same."""
    gp = _fit(_MEASUREMENTS)
    candidates = pd.DataFrame({"x": [0.25, 2.6, 4.9], "task": "T"})
    expected = gp.posterior(candidates).mean
    assert torch.equal(deepcopy(gp).posterior(candidates).mean, expected)
