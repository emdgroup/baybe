"""Tests for the Gaussian Process surrogate."""

import pytest
from gpytorch.kernels import MaternKernel as GPyTorchMaternKernel
from gpytorch.kernels import RBFKernel as GPyTorchRBFKernel
from gpytorch.kernels import ScaleKernel as GPyTorchScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood
from gpytorch.means import ConstantMean
from gpytorch.means import Mean as GPyTorchMean
from pandas.testing import assert_frame_equal
from pytest import param

from baybe.kernels.basic import MaternKernel, RBFKernel
from baybe.kernels.composite import ScaleKernel, SumKernel
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.surrogates.gaussian_process.components.generic import PlainGPComponentFactory
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.presets import GaussianProcessPreset
from baybe.targets.numerical import NumericalTarget
from baybe.utils.dataframe import create_fake_input

searchspace = NumericalContinuousParameter("p", (0, 1)).to_searchspace()
objective = NumericalTarget("t").to_objective()
measurements = create_fake_input(searchspace.parameters, objective.targets, n_rows=100)

baybe_kernel = ScaleKernel(SumKernel([MaternKernel(), RBFKernel()]))
gpytorch_kernel = GPyTorchScaleKernel(GPyTorchMaternKernel() + GPyTorchRBFKernel())


def _dummy_mean_factory(*args, **kwargs) -> GPyTorchMean:
    return ConstantMean()


def _dummy_likelihood_factory(*args, **kwargs) -> GPyTorchLikelihood:
    return GaussianLikelihood()


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
def test_gpytorch_components(component_1, component_2):
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
    kernel = GPyTorchMaternKernel()
    mean = ConstantMean()
    likelihood = GaussianLikelihood()

    # Works without overrides ...
    GaussianProcessSurrogate.from_preset(preset)

    # ... and with overrides
    gp = GaussianProcessSurrogate.from_preset(
        preset,
        kernel_or_factory=kernel,
        mean_or_factory=mean,
        likelihood_or_factory=likelihood,
    )
    assert isinstance(gp.kernel_factory, PlainGPComponentFactory)
    assert gp.kernel_factory.component is kernel
    assert isinstance(gp.mean_factory, PlainGPComponentFactory)
    assert gp.mean_factory.component is mean
    assert isinstance(gp.likelihood_factory, PlainGPComponentFactory)
    assert gp.likelihood_factory.component is likelihood
    gp.fit(searchspace, objective, measurements)


def test_invalid_components():
    """Passing invalid component types raises errors."""
    with pytest.raises(TypeError, match="Component must be one of"):
        GaussianProcessSurrogate(kernel_or_factory=ConstantMean())
    with pytest.raises(TypeError, match="Component must be one of"):
        GaussianProcessSurrogate(mean_or_factory=GaussianLikelihood())
    with pytest.raises(TypeError, match="Component must be one of"):
        GaussianProcessSurrogate(likelihood_or_factory=MaternKernel())
