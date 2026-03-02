"""Custom GPyTorch components."""

import torch
from botorch.models.multitask import _compute_multitask_mean
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods.hadamard_gaussian_likelihood import HadamardGaussianLikelihood
from gpytorch.means.multitask_mean import Mean, MultitaskMean
from gpytorch.priors import LogNormalPrior
from torch import Tensor
from torch.nn import Module


class HadamardConstantMean(Mean):
    """A GPyTorch mean function implementing BoTorch's multitask mean logic.

    Analogous to GPyTorch's
    https://github.com/cornellius-gp/gpytorch/blob/main/gpytorch/likelihoods/hadamard_gaussian_likelihood.py
    but where the logic is applied to the mean function, i.e. we learn a different
    (constant) mean for each task.
    """

    def __init__(self, mean_module: Module, num_tasks: int, task_feature: int):
        super().__init__()
        self.multitask_mean = MultitaskMean(mean_module, num_tasks=num_tasks)
        self.task_feature = task_feature

    def forward(self, x: Tensor) -> Tensor:
        # Convert task feature to positive index
        task_feature = self.task_feature % x.shape[-1]

        # Split input into task and non-task components
        x_before = x[..., :task_feature]
        task_idcs = x[..., task_feature : task_feature + 1]
        x_after = x[..., task_feature + 1 :]

        return _compute_multitask_mean(
            self.multitask_mean, x_before, task_idcs, x_after
        )


def make_botorch_multitask_likelihood(
    num_tasks: int, task_feature: int
) -> HadamardGaussianLikelihood:
    """Adapted from :class:`botorch.models.multitask.MultiTaskGP`."""
    noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
    return HadamardGaussianLikelihood(
        num_tasks=num_tasks,
        batch_shape=torch.Size(),
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(
            MIN_INFERRED_NOISE_LEVEL,
            transform=None,
            initial_value=noise_prior.mode,
        ),
        task_feature_index=task_feature,
    )
