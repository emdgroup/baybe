"""Gaussian process surrogates."""

from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
from attr import define, field
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import IndexKernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import GammaPrior
from torch import Tensor

from baybe.searchspace import SearchSpace
from baybe.surrogates.base import Surrogate
from baybe.surrogates.validation import get_model_params_validator


@define
class GaussianProcessSurrogate(Surrogate):
    """A Gaussian process surrogate model."""

    # Class variables
    joint_posterior: ClassVar[bool] = True
    # See base class.

    supports_transfer_learning: ClassVar[bool] = True
    # See base class.

    # Object variables
    model_params: Dict[str, Any] = field(
        factory=dict,
        converter=dict,
        validator=get_model_params_validator(SingleTaskGP.__init__),
    )
    # See base class.

    _model: Optional[SingleTaskGP] = field(init=False, default=None)
    """The actual model."""

    def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        # See base class.
        posterior = self._model.posterior(candidates)
        return posterior.mvn.mean, posterior.mvn.covariance_matrix

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        # See base class.

        # identify the indexes of the task and numeric dimensions
        # TODO: generalize to multiple task parameters
        task_idx = searchspace.task_idx
        n_task_params = 1 if task_idx else 0
        numeric_idxs = [i for i in range(train_x.shape[1]) if i != task_idx]

        # get the input bounds from the search space in BoTorch Format
        bounds = searchspace.param_bounds_comp
        # TODO: use target value bounds when explicitly provided

        # define the input and outcome transforms
        # TODO [Scaling]: scaling should be handled by search space object
        input_transform = Normalize(
            train_x.shape[1], bounds=bounds, indices=numeric_idxs
        )
        outcome_transform = Standardize(train_y.shape[1])

        # ---------- GP prior selection ---------- #
        # TODO: temporary prior choices adapted from edbo, replace later on

        mordred = searchspace.contains_mordred or searchspace.contains_rdkit
        if mordred and train_x.shape[-1] < 50:
            mordred = False

        # low D priors
        if train_x.shape[-1] < 5:
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
        mean_module = ConstantMean(batch_shape=batch_shape)

        # define the covariance module for the numeric dimensions
        base_covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=train_x.shape[-1] - n_task_params,
                active_dims=numeric_idxs,
                batch_shape=batch_shape,
                lengthscale_prior=lengthscale_prior[0],
            ),
            batch_shape=batch_shape,
            outputscale_prior=outputscale_prior[0],
        )
        base_covar_module.outputscale = torch.tensor([outputscale_prior[1]])
        base_covar_module.base_kernel.lengthscale = torch.tensor([lengthscale_prior[1]])

        # create GP covariance
        if task_idx is None:
            covar_module = base_covar_module
        else:
            task_covar_module = IndexKernel(
                num_tasks=searchspace.n_tasks,
                active_dims=task_idx,
                rank=searchspace.n_tasks,  # TODO: make controllable
            )
            covar_module = base_covar_module * task_covar_module

        # create GP likelihood
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior[0], batch_shape=batch_shape
        )
        likelihood.noise = torch.tensor([noise_prior[1]])

        # construct and fit the Gaussian process
        self._model = SingleTaskGP(
            train_x,
            train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
        )
        mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
        # IMPROVE: The step_limit=100 stems from the former (deprecated)
        #  `fit_gpytorch_torch` function, for which this was the default. Probably,
        #   one should use a smarter logic here.
        fit_gpytorch_mll_torch(mll, step_limit=100)
