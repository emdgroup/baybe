"""Gaussian process surrogates."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from attrs import define, field
from attrs.validators import instance_of
from sklearn.preprocessing import MinMaxScaler

from baybe.objective import Objective
from baybe.parameters.base import Parameter
from baybe.parameters.categorical import TaskParameter
from baybe.searchspace.core import SearchSpace
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
from baybe.utils.scaling import ScalerProtocol

if TYPE_CHECKING:
    from botorch.models.model import Model
    from botorch.posteriors import Posterior
    from torch import Tensor


@define
class _ModelContext:
    """Model context for :class:`GaussianProcessSurrogate`."""

    searchspace: SearchSpace = field(validator=instance_of(SearchSpace))
    """The search space the model is trained on."""

    @property
    def task_idx(self) -> int | None:
        """The computational column index of the task parameter, if available."""
        return self.searchspace.task_idx

    @property
    def is_multitask(self) -> bool:
        """Indicates if model is to be operated in a multi-task context."""
        return self.n_task_dimensions > 0

    @property
    def n_task_dimensions(self) -> int:
        """The number of task dimensions."""
        # TODO: Generalize to multiple task parameters
        return 1 if self.task_idx is not None else 0

    @property
    def n_tasks(self) -> int:
        """The number of tasks."""
        return self.searchspace.n_tasks

    @property
    def parameter_bounds(self) -> Tensor:
        """Get the search space parameter bounds in BoTorch Format."""
        import torch

        return torch.from_numpy(self.searchspace.comp_rep_bounds.values)

    def get_numerical_indices(self, n_inputs: int) -> list[int]:
        """Get the indices of the regular numerical model inputs."""
        return [i for i in range(n_inputs) if i != self.task_idx]


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

    def to_botorch(self) -> Model:  # noqa: D102
        # See base class.

        return self._model

    @staticmethod
    def _make_parameter_scaler(
        parameter: Parameter,
    ) -> ScalerProtocol | Literal["passthrough"]:
        # See base class.

        # Task parameters are handled separately through an index kernel
        if isinstance(parameter, TaskParameter):
            return "passthrough"

        return MinMaxScaler()

    @staticmethod
    def _get_model_context(
        searchspace: SearchSpace, objective: Objective
    ) -> _ModelContext:
        # See base class.
        return _ModelContext(searchspace=searchspace)

    def _posterior(self, candidates: Tensor, /) -> Posterior:
        # See base class.
        return self._model.posterior(candidates)

    def _fit(self, train_x: Tensor, train_y: Tensor, context: _ModelContext) -> None:
        # See base class.

        import botorch
        import gpytorch
        import torch

        numerical_idxs = context.get_numerical_indices(train_x.shape[-1])

        # extract the batch shape of the training data
        batch_shape = train_x.shape[:-2]

        # create GP mean
        mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)

        # define the covariance module for the numeric dimensions
        base_covar_module = self.kernel_factory(
            context.searchspace, train_x, train_y
        ).to_gpytorch(
            ard_num_dims=train_x.shape[-1] - context.n_task_dimensions,
            active_dims=numerical_idxs,
            batch_shape=batch_shape,
        )

        # create GP covariance
        if not context.is_multitask:
            covar_module = base_covar_module
        else:
            task_covar_module = gpytorch.kernels.IndexKernel(
                num_tasks=context.n_tasks,
                active_dims=context.task_idx,
                rank=context.n_tasks,  # TODO: make controllable
            )
            covar_module = base_covar_module * task_covar_module

        # create GP likelihood
        noise_prior = _default_noise_factory(context.searchspace, train_x, train_y)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior[0].to_gpytorch(), batch_shape=batch_shape
        )
        likelihood.noise = torch.tensor([noise_prior[1]])

        # construct and fit the Gaussian process
        self._model = botorch.models.SingleTaskGP(
            train_x,
            train_y,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
        )
        mll = gpytorch.ExactMarginalLogLikelihood(self._model.likelihood, self._model)
        botorch.fit_gpytorch_mll(mll)
