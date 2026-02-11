"""Tests for the Gaussian Process surrogate."""

import pytest
from gpytorch.kernels import MaternKernel as GPyTorchMaternKernel
from gpytorch.kernels import RBFKernel as GPyTorchRBFKernel
from gpytorch.kernels import ScaleKernel as GPyTorchScaleKernel
from pandas.testing import assert_frame_equal

from baybe.kernels.basic import MaternKernel, RBFKernel
from baybe.kernels.composite import AdditiveKernel, ScaleKernel
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.targets.numerical import NumericalTarget
from baybe.utils.dataframe import create_fake_input

searchspace = NumericalContinuousParameter("p", (0, 1)).to_searchspace()
objective = NumericalTarget("t").to_objective()


def test_gpytorch_kernel():
    """The GP accepts GPyTorch kernels and produces the same result as with BayBE kernels."""  # noqa: E501
    measurements = create_fake_input(searchspace.parameters, objective.targets)
    k1 = GPyTorchScaleKernel(GPyTorchMaternKernel() + GPyTorchRBFKernel())
    k2 = ScaleKernel(AdditiveKernel([MaternKernel(), RBFKernel()]))
    gp1 = GaussianProcessSurrogate(k1)
    gp2 = GaussianProcessSurrogate(k2)
    gp1.fit(searchspace, objective, measurements)
    gp2.fit(searchspace, objective, measurements)
    posterior1 = gp1.posterior_stats(measurements)
    posterior2 = gp2.posterior_stats(measurements)
    assert_frame_equal(posterior1, posterior2)


def test_gpytorch_kernel_serialization():
    """An error is raised when attempting to serialize a GP with a GPyTorch kernel."""
    with pytest.raises(NotImplementedError, match=".MaternKernel' is not supported."):
        GaussianProcessSurrogate(GPyTorchMaternKernel()).to_dict()
