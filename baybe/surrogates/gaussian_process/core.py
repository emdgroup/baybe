"""Gaussian process surrogates."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from attrs import define, field

from baybe.searchspace import SearchSpace
from baybe.surrogates.base import Surrogate
from baybe.surrogates.gaussian_process.kernel_factory import (
    KernelFactory,
    to_kernel_factory,
)
from baybe.surrogates.gaussian_process.presets import (
    GaussianProcessPreset,
    make_gp_from_preset,
)
from baybe.surrogates.gaussian_process.presets.default import (
    DefaultKernelFactory,
    _default_noise_factory,
)

if TYPE_CHECKING:
    from torch import Tensor


@define
class GaussianProcessSurrogate(Surrogate):
    """A Gaussian process surrogate model."""

    # Class variables
    joint_posterior: ClassVar[bool] = True
    # See base class.

    supports_transfer_learning: ClassVar[bool] = True
    # See base class.

    # Object variables
    kernel_factory: KernelFactory = field(
        alias="kernel_or_factory",
        factory=DefaultKernelFactory,
        converter=to_kernel_factory,
    )
    """The factory used to create the kernel of the Gaussian process.

    Accepts either a :class:`baybe.kernels.base.Kernel` or a
    :class:`.kernel_factory.KernelFactory`.
    When passing a :class:`baybe.kernels.base.Kernel`, it gets automatically wrapped
    into a :class:`.kernel_factory.PlainKernelFactory`."""

    # TODO: type should be Optional[botorch.models.SingleTaskGP] but is currently
    #   omitted due to: https://github.com/python-attrs/cattrs/issues/531
    _model = field(init=False, default=None, eq=False)
    """The actual model."""

    @classmethod
    def from_preset(preset: GaussianProcessPreset) -> GaussianProcessSurrogate:
        """Create a Gaussian process surrogate from one of the defined presets."""
        return make_gp_from_preset(preset)

    def _posterior(self, candidates: Tensor) -> tuple[Tensor, Tensor]:
        # See base class.
        posterior = self._model.posterior(candidates)
        return posterior.mvn.mean, posterior.mvn.covariance_matrix

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        # See base class.

        import botorch
        import gpytorch
        import torch

        # identify the indexes of the task and numeric dimensions
        # TODO: generalize to multiple task parameters
        task_idx = searchspace.task_idx
        n_task_params = 1 if task_idx is not None else 0
        numeric_idxs = [i for i in range(train_x.shape[1]) if i != task_idx]

        # get the input bounds from the search space in BoTorch Format
        bounds = torch.from_numpy(searchspace.param_bounds_comp)
        # TODO: use target value bounds when explicitly provided

        # define the input and outcome transforms
        # TODO [Scaling]: scaling should be handled by search space object
        input_transform = botorch.models.transforms.Normalize(
            train_x.shape[1], bounds=bounds, indices=numeric_idxs
        )
        outcome_transform = botorch.models.transforms.Standardize(train_y.shape[1])

        # extract the batch shape of the training data
        batch_shape = train_x.shape[:-2]

        # create GP mean
        mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)

        # define the covariance module for the numeric dimensions
        base_covar_module = self.kernel_factory(
            searchspace, train_x, train_y
        ).to_gpytorch(
            ard_num_dims=train_x.shape[-1] - n_task_params,
            active_dims=numeric_idxs,
            batch_shape=batch_shape,
        )

        # create GP covariance
        if task_idx is None:
            covar_module = base_covar_module
        else:
            task_covar_module = gpytorch.kernels.IndexKernel(
                num_tasks=searchspace.n_tasks,
                active_dims=task_idx,
                rank=searchspace.n_tasks,  # TODO: make controllable
            )
            covar_module = base_covar_module * task_covar_module

        # create GP likelihood
        noise_prior = _default_noise_factory(searchspace, train_x, train_y)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior[0].to_gpytorch(), batch_shape=batch_shape
        )
        likelihood.noise = torch.tensor([noise_prior[1]])

        # construct and fit the Gaussian process
        self._model = botorch.models.SingleTaskGP(
            train_x,
            train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
        )
        mll = gpytorch.ExactMarginalLogLikelihood(self._model.likelihood, self._model)
        botorch.fit_gpytorch_mll(mll)
