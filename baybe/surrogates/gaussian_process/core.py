"""Gaussian process surrogates."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, Union

from attrs import define, field
from attrs.validators import instance_of

from baybe.kernels import MaternKernel, ScaleKernel
from baybe.kernels.base import Kernel
from baybe.priors import GammaPrior
from baybe.searchspace import SearchSpace
from baybe.serialization.core import (
    converter,
    get_base_structure_hook,
    unstructure_base,
)
from baybe.serialization.mixin import SerialMixin
from baybe.surrogates.base import Surrogate

if TYPE_CHECKING:
    from torch import Tensor


class KernelFactory(Protocol):
    """A protocol defining the interface expected for kernel factories."""

    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        """Create a :class:`baybe.kernels.base.Kernel` for the given context."""
        ...


# Register de-/serialization hooks
converter.register_structure_hook(KernelFactory, get_base_structure_hook(KernelFactory))
converter.register_unstructure_hook(KernelFactory, unstructure_base)


@define(frozen=True)
class PlainKernelFactory(KernelFactory, SerialMixin):
    """A trivial factory that returns a fixed pre-defined kernel upon request."""

    kernel: Kernel = field(validator=instance_of(Kernel))

    def __call__(  # noqa: D102
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        # See base class.

        return self.kernel


def to_kernel_factory(x: Union[Kernel, KernelFactory], /) -> KernelFactory:
    """Wrap a kernel into a plain kernel factory (with factory passthrough)."""
    return PlainKernelFactory(x) if isinstance(x, Kernel) else x


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
        factory=lambda: PlainKernelFactory(ScaleKernel(MaternKernel())),
        converter=to_kernel_factory,
    )
    """The factory used to create the kernel of the Gaussian process.
    When passing a :class:`baybe.kernels.base.Kernel`, it gets automatically wrapped
    into a :class:`baybe.surrogates.gaussian_process.core.PlainKernelFactory`."""

    # TODO: type should be Optional[botorch.models.SingleTaskGP] but is currently
    #   omitted due to: https://github.com/python-attrs/cattrs/issues/531
    _model = field(init=False, default=None, eq=False)
    """The actual model."""

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

        # ---------- GP prior selection ---------- #
        # TODO: temporary prior choices adapted from edbo, replace later on

        mordred = (searchspace.contains_mordred or searchspace.contains_rdkit) and (
            train_x.shape[-1] >= 50
        )

        # TODO Until now, only the kernels use our custom priors, hence the explicit
        # to_gpytorch() calls for all others
        # low D priors
        if train_x.shape[-1] < 10:
            lengthscale_prior = [GammaPrior(1.2, 1.1), 0.2]
            outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
            noise_prior = [GammaPrior(1.05, 0.5), 0.1]

        # DFT optimized priors
        elif mordred and train_x.shape[-1] < 100:
            lengthscale_prior = [GammaPrior(2.0, 0.2), 5.0]
            outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
            noise_prior = [GammaPrior(1.5, 0.1), 5.0]

        # Mordred optimized priors
        elif mordred:
            lengthscale_prior = [GammaPrior(2.0, 0.1), 10.0]
            outputscale_prior = [GammaPrior(2.0, 0.1), 10.0]
            noise_prior = [GammaPrior(1.5, 0.1), 5.0]

        # OHE optimized priors
        else:
            lengthscale_prior = [GammaPrior(3.0, 1.0), 2.0]
            outputscale_prior = [GammaPrior(5.0, 0.2), 20.0]
            noise_prior = [GammaPrior(1.5, 0.1), 5.0]

        # ---------- End: GP prior selection ---------- #

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
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior[0].to_gpytorch(), batch_shape=batch_shape
        )
        if noise_prior[1] is not None:
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
