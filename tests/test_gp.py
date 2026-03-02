"""Tests for the Gaussian Process surrogate."""

import pandas as pd
import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel as GPyTorchMaternKernel
from gpytorch.kernels import RBFKernel as GPyTorchRBFKernel
from gpytorch.kernels import ScaleKernel as GPyTorchScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood
from gpytorch.means import ConstantMean
from gpytorch.means import Mean as GPyTorchMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from pandas.testing import assert_frame_equal
from pytest import param

from baybe import active_settings
from baybe.kernels.basic import MaternKernel, RBFKernel
from baybe.kernels.composite import AdditiveKernel, ScaleKernel
from baybe.parameters.categorical import TaskParameter
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.presets import GaussianProcessPreset
from baybe.targets.numerical import NumericalTarget
from baybe.utils.dataframe import create_fake_input, to_tensor

searchspace = NumericalContinuousParameter("p", (0, 1)).to_searchspace()
searchspace_mt = SearchSpace.from_product(
    [
        NumericalContinuousParameter("p", (0, 1)),
        TaskParameter("task", ["a", "b", "c"]),
    ]
)
objective = NumericalTarget("t").to_objective()
measurements = create_fake_input(searchspace.parameters, objective.targets, n_rows=100)
measurements_mt = create_fake_input(
    searchspace_mt.parameters, objective.targets, n_rows=100
)
baybe_kernel = ScaleKernel(AdditiveKernel([MaternKernel(), RBFKernel()]))
gpytorch_kernel = GPyTorchScaleKernel(GPyTorchMaternKernel() + GPyTorchRBFKernel())


def _dummy_mean_factory(*args, **kwargs) -> GPyTorchMean:
    return ConstantMean()


def _dummy_likelihood_factory(*args, **kwargs) -> GPyTorchLikelihood:
    return GaussianLikelihood()


def _posterior_stats_botorch(
    searchspace: SearchSpace, measurements: pd.DataFrame
) -> pd.DataFrame:
    """The essential BoTorch stesp to produce posterior estimates."""
    train_X = to_tensor(searchspace.transform(measurements, allow_extra=True))
    train_Y = to_tensor(objective.transform(measurements, allow_extra=True))

    # >>>>> Code adapted from BoTorch landing page: https://botorch.org/ >>>>>
    # NOTE:
    # We normalize according to the searchspace bounds to ensure consisteny with
    # the BayBE GP implementation.
    if searchspace.n_tasks == 1:
        gp = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=Normalize(
                d=len(searchspace.comp_rep_columns),
                bounds=to_tensor(searchspace.scaling_bounds),
            ),
            outcome_transform=Standardize(m=1),
        )
    else:
        assert searchspace.task_idx is not None
        gp = MultiTaskGP(
            train_X=train_X, train_Y=train_Y, task_feature=searchspace.task_idx
        )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    # <<<<<<<<<<

    with torch.no_grad():
        posterior = gp.posterior(train_X)
    mean = posterior.mean
    std = posterior.variance.sqrt()
    return pd.DataFrame({"t_mean": mean.numpy().ravel(), "t_std": std.numpy().ravel()})


@pytest.mark.parametrize(
    ("component_1", "component_2"),
    [
        param(
            {"kernel_or_factory": baybe_kernel},
            {"kernel_or_factory": gpytorch_kernel},
            id="kernel",
        ),
        param(
            {"mean_or_factory": ConstantMean()},
            {"mean_or_factory": _dummy_mean_factory},
            id="mean",
        ),
        param(
            {"likelihood_or_factory": GaussianLikelihood()},
            {"likelihood_or_factory": _dummy_likelihood_factory},
            id="likelihood",
        ),
    ],
)
def test_gpytorch_kernel(component_1, component_2):
    """The GP accepts GPyTorch components and produces consistent results."""
    gp1 = GaussianProcessSurrogate(**component_1)
    gp2 = GaussianProcessSurrogate(**component_2)
    gp1.fit(searchspace, objective, measurements)
    gp2.fit(searchspace, objective, measurements)
    posterior1 = gp1.posterior_stats(measurements)
    posterior2 = gp2.posterior_stats(measurements)
    assert_frame_equal(posterior1, posterior2)


@pytest.mark.parametrize(
    "component",
    [
        param({"kernel_or_factory": GPyTorchMaternKernel()}, id="kernel"),
        param({"mean_or_factory": ConstantMean()}, id="mean"),
        param({"likelihood_or_factory": GaussianLikelihood()}, id="likelihood"),
    ],
)  # noqa: E501
def test_gpytorch_component_serialization(component):
    """An error is raised when attempting to serialize a GP with a GPyTorch component."""  # noqa: E501
    obj = next(iter(component.values()))
    msg = f"{type(obj).__name__}' is not supported"
    with pytest.raises(NotImplementedError, match=msg):
        GaussianProcessSurrogate(**component).to_dict()


@pytest.mark.parametrize("preset", list(GaussianProcessPreset), ids=lambda p: p.name)
def test_presets(preset: GaussianProcessPreset):
    """Presets can be loaded and their defaults can be overridden."""
    GaussianProcessSurrogate.from_preset(
        preset, GPyTorchMaternKernel(), ConstantMean(), GaussianLikelihood()
    )


@pytest.mark.parametrize("multitask", [False, True], ids=["single-task", "multi-task"])
def test_botorch_preset(multitask: bool, monkeypatch):
    """The BoTorch preset exactly mimics BoTorch's behavior."""
    if multitask:
        monkeypatch.setenv("BAYBE_DISABLE_CUSTOM_KERNEL_WARNING", "true")
        sp = searchspace_mt
        data = measurements_mt
    else:
        sp = searchspace
        data = measurements

    active_settings.random_seed = 1337
    gp = GaussianProcessSurrogate.from_preset("BOTORCH")
    gp.fit(sp, objective, data)
    posterior1 = gp.posterior_stats(data)

    active_settings.random_seed = 1337
    posterior2 = _posterior_stats_botorch(sp, data)

    assert_frame_equal(posterior1, posterior2)
