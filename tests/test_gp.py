"""Tests for the Gaussian Process surrogate."""

import sys

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
from baybe.kernels.composite import ScaleKernel
from baybe.parameters.categorical import TaskParameter
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.components.fit_criterion import FitCriterion
from baybe.surrogates.gaussian_process.components.generic import PlainGPComponentFactory
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
baybe_kernel = ScaleKernel(MaternKernel() + RBFKernel())
gpytorch_kernel = GPyTorchScaleKernel(GPyTorchMaternKernel() + GPyTorchRBFKernel())


def _dummy_mean_factory(*args, **kwargs) -> GPyTorchMean:
    return ConstantMean()


def _dummy_likelihood_factory(*args, **kwargs) -> GPyTorchLikelihood:
    return GaussianLikelihood()


def _posterior_stats_botorch(
    searchspace: SearchSpace, measurements: pd.DataFrame
) -> pd.DataFrame:
    """The essential BoTorch steps to produce posterior estimates."""
    train_X = to_tensor(searchspace.transform(measurements, allow_extra=True))
    train_Y = to_tensor(objective.transform(measurements, allow_extra=True))

    # >>>>> Code adapted from BoTorch landing page: https://botorch.org/ >>>>>
    # NOTE: We normalize according to the searchspace bounds to ensure consistency with
    #       the BayBE GP implementation.
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
        non_task_idcs = [
            i for i in range(train_X.shape[-1]) if i != searchspace.task_idx
        ]
        gp = MultiTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=searchspace.task_idx,
            input_transform=Normalize(
                d=len(searchspace.comp_rep_columns),
                indices=non_task_idcs,
                bounds=to_tensor(searchspace.scaling_bounds),
            ),
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
    criterion = FitCriterion.LEAVE_ONE_OUT_PSEUDOLIKELIHOOD

    # Works without overrides ...
    gp1 = GaussianProcessSurrogate.from_preset(preset)

    # ... and with overrides
    gp2 = GaussianProcessSurrogate.from_preset(
        preset,
        kernel_or_factory=kernel,
        mean_or_factory=mean,
        likelihood_or_factory=likelihood,
        fit_criterion_or_factory=criterion,
    )

    # Check that the overrides were applied correctly
    assert isinstance(gp2.kernel_factory, PlainGPComponentFactory)
    assert gp2.kernel_factory.component is kernel
    assert isinstance(gp2.mean_factory, PlainGPComponentFactory)
    assert gp2.mean_factory.component is mean
    assert isinstance(gp2.likelihood_factory, PlainGPComponentFactory)
    assert gp2.likelihood_factory.component is likelihood
    assert isinstance(gp2.fit_criterion_factory, PlainGPComponentFactory)
    assert gp2.fit_criterion_factory.component == criterion
    assert gp2.fit_criterion_factory != gp1.fit_criterion_factory

    gp2.fit(searchspace, objective, measurements)


def test_invalid_components():
    """Passing invalid component types raises errors."""
    with pytest.raises(TypeError, match="Component must be one of"):
        GaussianProcessSurrogate(kernel_or_factory=ConstantMean())
    with pytest.raises(TypeError, match="Component must be one of"):
        GaussianProcessSurrogate(mean_or_factory=GaussianLikelihood())
    with pytest.raises(TypeError, match="Component must be one of"):
        GaussianProcessSurrogate(
            likelihood_or_factory=FitCriterion.LEAVE_ONE_OUT_PSEUDOLIKELIHOOD
        )
    with pytest.raises(TypeError, match="Component must be one of"):
        GaussianProcessSurrogate(fit_criterion_or_factory=MaternKernel())


# NOTE: The BOTORCH preset tracks BoTorch's GP defaults while the HVARFNER preset
#   implements BoTorch's static Hvarfner et al. (2024) parametrization. Therefore, the
#   presets diverge as BoTorch evolves (e.g., BetaPrior added in 0.18.0).
@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="BoTorch >=0.18.0 requires Python >=3.11.",
)
@pytest.mark.parametrize("multitask", [False, True], ids=["single-task", "multi-task"])
def test_botorch_preset(multitask: bool):
    """The BoTorch preset exactly mimics BoTorch's MultiTaskGP/SingleTaskGP behavior."""
    if multitask:
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


def test_get_posterior_mean_correct_under_different_bounds():
    """Posterior mean evaluates at correct physical points when bounds differ."""
    from baybe.parameters.numerical import NumericalDiscreteParameter

    # Train a surrogate on a narrow search space [0, 5]
    prior_params = [NumericalDiscreteParameter("x1", values=[0.0, 2.5, 5.0])]
    prior_ss = SearchSpace.from_product(prior_params)
    prior_obj = NumericalTarget(name="y").to_objective()

    prior_surrogate = GaussianProcessSurrogate()
    prior_meas = pd.DataFrame({"x1": [0.0, 2.5, 5.0], "y": [0.0, 5.0, 10.0]})
    prior_surrogate.fit(prior_ss, prior_obj, prior_meas)

    # Get the surrogate's prediction at x1=2.5
    expected_mean = prior_surrogate.posterior(pd.DataFrame({"x1": [2.5]})).mean.item()

    # New GP on a WIDER search space [0, 10], using the get_posterior_mean method
    new_params = [NumericalDiscreteParameter("x1", values=[0.0, 2.5, 5.0, 7.5, 10.0])]
    new_ss = SearchSpace.from_product(new_params)

    new_surrogate = GaussianProcessSurrogate(
        mean_or_factory=prior_surrogate.get_posterior_mean
    )
    # Train on data that lies exactly on the prior mean to avoid kernel effects
    training_points = pd.DataFrame({"x1": [0.0, 10.0]})
    with torch.no_grad():
        training_targets = prior_surrogate.posterior(training_points).mean
    new_meas = pd.DataFrame(
        {
            "x1": training_points["x1"],
            "y": training_targets.numpy().ravel(),
        }
    )
    new_surrogate.fit(new_ss, prior_obj, new_meas)

    # Test end-to-end: the posterior should match the prior mean
    actual_mean = new_surrogate.posterior(pd.DataFrame({"x1": [2.5]})).mean.item()

    assert abs(actual_mean - expected_mean) < 1e-4


def test_get_posterior_mean_same_bounds():
    """Posterior mean is correct when both search spaces have the same bounds."""
    from baybe.parameters.numerical import NumericalDiscreteParameter

    params = [NumericalDiscreteParameter("x1", values=[0.0, 2.5, 5.0])]
    ss = SearchSpace.from_product(params)
    obj = NumericalTarget(name="y").to_objective()

    prior_surrogate = GaussianProcessSurrogate()
    meas = pd.DataFrame({"x1": [0.0, 2.5, 5.0], "y": [0.0, 5.0, 10.0]})
    prior_surrogate.fit(ss, obj, meas)

    expected_mean = prior_surrogate.posterior(pd.DataFrame({"x1": [2.5]})).mean.item()

    new_surrogate = GaussianProcessSurrogate(
        mean_or_factory=prior_surrogate.get_posterior_mean
    )
    # Train on data that lies exactly on the prior mean
    training_points = pd.DataFrame({"x1": [0.0, 5.0]})
    with torch.no_grad():
        training_targets = prior_surrogate.posterior(training_points).mean
    new_meas = pd.DataFrame(
        {
            "x1": training_points["x1"],
            "y": training_targets.numpy().ravel(),
        }
    )
    new_surrogate.fit(ss, obj, new_meas)

    # Test end-to-end: the posterior should match the prior mean
    actual_mean = new_surrogate.posterior(pd.DataFrame({"x1": [2.5]})).mean.item()

    assert abs(actual_mean - expected_mean) < 1e-4


def test_get_posterior_mean_raises_if_not_fitted():
    """Calling get_posterior_mean raises if the surrogate has not been fitted."""
    from baybe.exceptions import ModelNotTrainedError

    with pytest.raises(ModelNotTrainedError, match="must be fitted"):
        GaussianProcessSurrogate().get_posterior_mean(
            searchspace, objective, measurements
        )
